"""Connect journal; protocol snapshots are projected into SDK DatabaseTaskStore.

A durable projection in the same transaction as input acceptance closes the
cross-database crash window. Startup replays projections without running science.
"""

import json
import time
from contextlib import asynccontextmanager

import aiosqlite


class Store:
    def __init__(self, path):
        self.path = path
        self.db = None

    async def open(self):
        self.db = await aiosqlite.connect(self.path)
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=FULL;
            PRAGMA busy_timeout=5000;
            CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);
            INSERT INTO schema_version SELECT 2 WHERE NOT EXISTS (SELECT 1 FROM schema_version);
            CREATE TABLE IF NOT EXISTS contexts(
                owner TEXT, context_id TEXT, thread_id TEXT, profile TEXT,
                PRIMARY KEY(owner, context_id));
            CREATE TABLE IF NOT EXISTS records(
                task_id TEXT PRIMARY KEY, owner TEXT NOT NULL, context_id TEXT NOT NULL,
                phase TEXT NOT NULL, updated REAL NOT NULL, record TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS messages(
                owner TEXT, message_id TEXT, digest TEXT NOT NULL, task_id TEXT NOT NULL,
                PRIMARY KEY(owner, message_id));
            CREATE TABLE IF NOT EXISTS operations(
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS artifacts(
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS outbox(
                id TEXT PRIMARY KEY, task_id TEXT NOT NULL, state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, next_attempt REAL NOT NULL DEFAULT 0,
                claimed_at REAL, receipt TEXT, payload TEXT NOT NULL);
        """)
        version = await self.one("SELECT version FROM schema_version")
        if version["version"] != 2:
            raise RuntimeError("Unsupported service storage schema; use fresh integrated-service state")

    async def close(self):
        if self.db:
            await self.db.close()

    async def one(self, sql, params=()):
        async with self.db.execute(sql, params) as cursor:
            return await cursor.fetchone()

    async def all(self, sql, params=()):
        async with self.db.execute(sql, params) as cursor:
            return await cursor.fetchall()

    @asynccontextmanager
    async def transaction(self):
        # Service's lock serializes transactions and callers on this connection.
        await self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            await self.db.commit()
        except BaseException:
            await self.db.rollback()
            raise

    async def put(self, record):
        await self.db.execute(
            "INSERT OR REPLACE INTO records VALUES(?,?,?,?,?,?)",
            (
                record["id"],
                record["owner"],
                record["context_id"],
                record["phase"],
                time.time(),
                json.dumps(record),
            ),
        )

    async def get(self, task_id):
        row = await self.one("SELECT record FROM records WHERE task_id=?", (task_id,))
        return json.loads(row["record"]) if row else None

    async def records(self):
        return [
            json.loads(r["record"])
            for r in await self.all("SELECT record FROM records ORDER BY updated, task_id")
        ]
