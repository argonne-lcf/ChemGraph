"""Transactional operation records for single-host embedding services."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3

from chemgraph.runtime.journal import OperationConflict


@contextmanager
def connection(path):
    """Open a short-lived connection; callers never share one across threads."""
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA synchronous=FULL")
        with db:
            yield db
    finally:
        db.close()


def initialize_database(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.close(fd)
    with connection(path) as db:
        db.execute("PRAGMA journal_mode=WAL")
    return path


class SqliteOperationStore:
    """OperationStore with durable CAS writes and optional owner-bound access.

    An unbound store is for trusted administration/recovery only. An owner view
    enforces ownership on get, put and records, including thread-filtered lists.
    SQLite WAL requires a local filesystem. It must not be placed on NFS.
    """

    def __init__(self, path, *, owner=None):
        self.path = initialize_database(path)
        self.owner = owner
        with connection(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, thread_id TEXT NOT NULL,
                state TEXT NOT NULL, revision INTEGER NOT NULL, record TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS operations_owner_state ON operations(owner, state)")
            db.execute("CREATE INDEX IF NOT EXISTS operations_thread ON operations(thread_id)")

    def for_owner(self, owner):
        if not owner or (self.owner is not None and self.owner != owner):
            raise PermissionError("Invalid operation owner.")
        view = object.__new__(type(self))
        view.path, view.owner = self.path, owner
        return view

    async def get(self, identifier):
        def read():
            with connection(self.path) as db:
                row = db.execute("SELECT owner, record FROM operations WHERE id=?", (identifier,)).fetchone()
            if row is None or (self.owner is not None and row["owner"] != self.owner):
                return None
            return json.loads(row["record"])
        return await asyncio.to_thread(read)

    async def put(self, record, *, expected_revision):
        if self.owner is not None and record["owner"] != self.owner:
            raise PermissionError("Invalid operation owner.")
        if record["revision"] != (expected_revision or 0) + 1:
            raise OperationConflict("Invalid operation revision.")
        payload = json.dumps(record, allow_nan=False)

        def write():
            with connection(self.path) as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute("SELECT owner, thread_id, revision FROM operations WHERE id=?",
                                 (record["id"],)).fetchone()
                if ((row is None and expected_revision is not None)
                        or (row is not None and (row["revision"] != expected_revision
                            or row["owner"] != record["owner"]
                            or row["thread_id"] != record["thread_id"]))):
                    raise OperationConflict("Operation changed concurrently.")
                db.execute("""INSERT INTO operations VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET state=excluded.state,
                    revision=excluded.revision, record=excluded.record""",
                    (record["id"], record["owner"], record["thread_id"], record["state"],
                     record["revision"], payload))
        await asyncio.to_thread(write)

    async def records(self, *, thread_id=None, state=None):
        def read():
            conditions, values = [], []
            for field, value in (("owner", self.owner), ("thread_id", thread_id), ("state", state)):
                if value is not None:
                    conditions.append(f"{field}=?")
                    values.append(value)
            query = "SELECT record FROM operations"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            with connection(self.path) as db:
                rows = db.execute(query + " ORDER BY rowid", values).fetchall()
            return [json.loads(row[0]) for row in rows]
        return await asyncio.to_thread(read)
