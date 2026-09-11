"""Persistent HTTP resources and an append-only, replayable event stream."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import shutil
import time
from uuid import uuid4


TERMINAL = {"completed", "failed", "interrupted", "cancelled"}
ACTIVE = {"queued", "running", "waiting_for_input", "cancelling"}


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
                CREATE TABLE IF NOT EXISTS upload_reservations (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                    bytes INTEGER NOT NULL
                );
            """)
            for table, columns in {
                "sessions": {
                    "provenance": "TEXT",
                    "last_activity": "REAL NOT NULL DEFAULT 0",
                },
                "runs": {"error_code": "TEXT", "waiting_since": "REAL"},
            }.items():
                existing = {
                    row["name"] for row in db.execute(f"PRAGMA table_info({table})")
                }
                for name, definition in columns.items():
                    if name not in existing:
                        db.execute(
                            f"ALTER TABLE {table} ADD COLUMN {name} {definition}"
                        )
            # Older deployments did not record activity. Give that history a
            # full retention window after upgrade rather than expiring it based
            # on conversation creation, which may predate recent follow-ups.
            db.execute(
                "UPDATE sessions SET last_activity=? WHERE last_activity=0",
                (time.time(),),
            )
            # Earlier workers published every JSON file, including agent state.
            # Revoke those download IDs while retaining diagnostics on disk.
            db.execute(
                "DELETE FROM files WHERE kind!='upload' AND name GLOB 'state_thread_*.json'"
            )
            db.executescript("""
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_web_run_v2 ON runs(session_id)
                    WHERE status IN ('queued','running','waiting_for_input','cancelling');
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

    @contextmanager
    def transaction(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            yield db

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

    def finish(
        self, run_id, status, *, text="", error=None, error_code=None, stopped=False
    ):
        with self.transaction() as db:
            if stopped:
                current = db.execute(
                    "SELECT status FROM runs WHERE id=?", (run_id,)
                ).fetchone()
                if current and current["status"] == "cancelling":
                    status, error_code, error = (
                        "cancelled",
                        "cancelled",
                        "The run was cancelled.",
                    )
            changed = db.execute(
                "UPDATE runs SET status=?, final_text=?, error=?, error_code=?, question_id=NULL, "
                "question=NULL,waiting_since=NULL WHERE id=? AND status IN ('queued','running','waiting_for_input'"
                + (",'cancelling'" if status == "cancelled" else "")
                + ")",
                (status, text, error, error_code, run_id),
            ).rowcount
            if changed:
                db.execute(
                    "INSERT INTO events(run_id,type,data) VALUES (?,?,?)",
                    (run_id, "status", json.dumps({"status": status})),
                )
                db.execute(
                    "UPDATE sessions SET last_activity=? WHERE id=(SELECT session_id FROM runs WHERE id=?)",
                    (time.time(), run_id),
                )

    def recover(self):
        for row in self.rows(
            "SELECT id,status FROM runs WHERE status IN ('queued','running','waiting_for_input','cancelling')"
        ):
            self.finish(
                row["id"],
                "cancelled" if row["status"] == "cancelling" else "interrupted",
                error="The server restarted. You can retry this request.",
                error_code="server_restarted",
            )
        self.execute("DELETE FROM upload_reservations")
        staging = self.root / "staging"
        if staging.exists():
            shutil.rmtree(staging)

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

    def owner_usage(self, owner, db=None):
        """Account for workspace files, memory, transcripts, and reserved uploads."""
        if db is None:
            with self.connect() as connection:
                return self.owner_usage(owner, connection)
        total = 0
        for row in db.execute("SELECT id FROM sessions WHERE owner=?", (owner,)):
            session_id = row["id"]
            paths = list(self.session_dir(session_id).rglob("*"))
            paths.extend((self.root / "diagnostics" / session_id).rglob("*"))
            paths.extend((self.root / "memory").glob(f"{session_id}.db*"))
            for path in paths:
                try:
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
                except FileNotFoundError:
                    pass
        total += db.execute(
            "SELECT COALESCE(SUM(u.bytes),0) FROM upload_reservations u JOIN sessions s ON s.id=u.session_id WHERE s.owner=?",
            (owner,),
        ).fetchone()[0]
        total += db.execute(
            "SELECT COALESCE(SUM(length(CAST(r.query || r.final_text || COALESCE(r.question,'') || COALESCE(r.answer,'') AS BLOB))),0) FROM runs r JOIN sessions s ON s.id=r.session_id WHERE s.owner=?",
            (owner,),
        ).fetchone()[0]
        total += db.execute(
            "SELECT COALESCE(SUM(length(CAST(e.data AS BLOB))),0) FROM events e JOIN runs r ON r.id=e.run_id JOIN sessions s ON s.id=r.session_id WHERE s.owner=?",
            (owner,),
        ).fetchone()[0]
        return total

    def cleanup(self, retention_days):
        """Remove inactive conversations, including files and agent memory."""
        cutoff = time.time() - retention_days * 86400
        with self.transaction() as db:
            rows = db.execute(
                "SELECT id FROM sessions s WHERE last_activity<? AND NOT EXISTS (SELECT 1 FROM runs r WHERE r.session_id=s.id AND r.status IN ('queued','running','waiting_for_input','cancelling')) AND NOT EXISTS (SELECT 1 FROM upload_reservations u WHERE u.session_id=s.id)",
                (cutoff,),
            ).fetchall()
            for row in rows:
                session_id = row["id"]
                workspace = self.session_dir(session_id)
                if workspace.exists():
                    shutil.rmtree(workspace)
                diagnostics = self.root / "diagnostics" / session_id
                if diagnostics.exists():
                    shutil.rmtree(diagnostics)
                for path in (self.root / "memory").glob(f"{session_id}.db*"):
                    path.unlink(missing_ok=True)
                db.execute("DELETE FROM files WHERE session_id=?", (session_id,))
                db.execute(
                    "DELETE FROM events WHERE run_id IN (SELECT id FROM runs WHERE session_id=?)",
                    (session_id,),
                )
                db.execute("DELETE FROM runs WHERE session_id=?", (session_id,))
                db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return len(rows)
