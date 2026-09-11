"""Persistent HTTP resources and an append-only, replayable event stream."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


TERMINAL = {"completed", "failed", "interrupted"}


class Store:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db_path = self.root / "web.db"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, model TEXT NOT NULL,
                    workflow TEXT NOT NULL, title TEXT NOT NULL,
                    created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    query TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
                    final_text TEXT NOT NULL DEFAULT '', error TEXT,
                    question_id TEXT, question TEXT, answer TEXT,
                    request_id TEXT NOT NULL, attachments TEXT NOT NULL DEFAULT '[]',
                    created TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    UNIQUE(session_id, request_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_run ON runs(session_id)
                    WHERE status IN ('queued', 'running', 'waiting_for_input');
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id), type TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS run_events ON events(run_id, id);
                CREATE TABLE IF NOT EXISTS files (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    run_id TEXT REFERENCES runs(id), name TEXT NOT NULL,
                    path TEXT NOT NULL, kind TEXT NOT NULL, size INTEGER NOT NULL
                );
            """)
        self.db_path.chmod(0o600)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def rows(self, sql, args=()):
        with self.connect() as db:
            return [dict(row) for row in db.execute(sql, args)]

    def one(self, sql, args=()):
        rows = self.rows(sql, args)
        return rows[0] if rows else None

    def execute(self, sql, args=()):
        with self.connect() as db:
            return db.execute(sql, args).rowcount

    def session_dir(self, session_id):
        # IDs originate in this store, never from a filesystem path supplied by a client.
        return self.root / "sessions" / session_id

    def event(self, run_id, event_type, data):
        self.execute(
            "INSERT INTO events(run_id,type,data) VALUES (?,?,?)",
            (run_id, event_type, json.dumps(data)),
        )

    def finish(self, run_id, status, *, text="", error=None):
        with self.connect() as db:
            changed = db.execute(
                "UPDATE runs SET status=?, final_text=?, error=?, question_id=NULL, "
                "question=NULL WHERE id=? AND status IN ('queued','running','waiting_for_input')",
                (status, text, error, run_id),
            ).rowcount
            if changed:
                db.execute(
                    "INSERT INTO events(run_id,type,data) VALUES (?,?,?)",
                    (run_id, "status", json.dumps({"status": status})),
                )

    def recover(self):
        for row in self.rows(
            "SELECT id FROM runs WHERE status IN ('queued','running','waiting_for_input')"
        ):
            self.finish(
                row["id"],
                "interrupted",
                error="The server restarted. You can retry this request.",
            )

    def register_file(self, session_id, path, kind, run_id=None, name=None):
        path = Path(path).resolve()
        path.relative_to(self.session_dir(session_id).resolve())
        file_id = str(uuid4())
        self.execute(
            "INSERT INTO files VALUES (?,?,?,?,?,?,?)",
            (
                file_id,
                session_id,
                run_id,
                name or path.name,
                str(path.relative_to(self.root)),
                kind,
                path.stat().st_size,
            ),
        )
        return file_id

    def file_path(self, record):
        path = (self.root / record["path"]).resolve()
        path.relative_to(self.session_dir(record["session_id"]).resolve())
        if not path.is_file():
            raise FileNotFoundError(record["id"])
        return path
