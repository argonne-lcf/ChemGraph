"""Private service records and an exclusive process lease for one data directory."""

import json
import sqlite3
from pathlib import Path


class ServiceStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lease = sqlite3.connect(self.root / "lease.db", timeout=0)
        try:
            self.lease.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            self.lease.close()
            raise RuntimeError("Another RunService owns this data directory.") from exc
        self.connection = sqlite3.connect(self.root / "service.db")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS records (kind TEXT, id TEXT, body TEXT, PRIMARY KEY(kind,id))")
        self.connection.commit()

    def put(self, kind, record):
        value = record.model_dump(mode="json")
        if kind == "run":
            value["responses"] = record.responses
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO records VALUES (?,?,?)",
                                    (kind, record.id, json.dumps(value, allow_nan=False)))

    def get(self, kind, identifier):
        row = self.connection.execute("SELECT body FROM records WHERE kind=? AND id=?",
                                      (kind, identifier)).fetchone()
        if row is None:
            raise KeyError(identifier)
        return json.loads(row[0])

    def records(self, kind):
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT body FROM records WHERE kind=? ORDER BY rowid", (kind,))]

    def close(self):
        self.connection.close()
        self.lease.close()
