"""Private workspaces, bounded ingestion, and immutable owner-scoped artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
from types import SimpleNamespace
import uuid

from chemgraph.runtime.artifacts import (
    INTERNAL_DIRS, artifact_file, capture_artifacts, file_identity, output_manifest, verify_file,
)
from chemgraph.runtime.context import WorkspaceScope, scoped_path, workspace_scope
from chemgraph.runtime.sqlite_store import connection


class QuotaExceeded(ValueError):
    """The request exceeds an operator-configured byte or storage limit."""


def relative_path(path):
    if (not isinstance(path, str) or not path or "\\" in path or "\x00" in path
            or any(ord(char) < 32 for char in path)
            or Path(path).is_absolute() or PureWindowsPath(path).drive
            or ".." in Path(path).parts or INTERNAL_DIRS.intersection(Path(path).parts)
            or any(part.startswith(".service-") for part in Path(path).parts)
            or Path(path) == Path(".")):
        raise ValueError("Use a workspace-relative file path without reserved components.")
    return path


class Files:
    """Mutating methods run under the service's per-owner lock."""

    def __init__(self, config):
        self.config = config
        with connection(config.database) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, operation_id TEXT NOT NULL,
                workspace TEXT NOT NULL, descriptor TEXT NOT NULL
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS artifacts_owner ON artifacts(owner)")

    def workspace(self, owner):
        return self.config.data_dir / "workspaces" / owner

    def job_workspace(self, owner, identifier):
        return self.config.data_dir / "jobs" / owner / identifier

    def path(self, owner, path):
        root = self.workspace(owner)
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with workspace_scope(WorkspaceScope(owner, root)):
            return Path(scoped_path(relative_path(path)))

    def usage(self, owner):
        size = 0
        for root in (self.workspace(owner), self.config.data_dir / "jobs" / owner):
            for path in root.rglob("*") if root.exists() else ():
                try:
                    if path.is_file() and not path.is_symlink():
                        size += path.stat().st_size
                except FileNotFoundError:
                    pass  # A worker may finish cleaning a temporary file during the scan.
        return size

    def check_quota(self, owner, additional=0):
        if self.usage(owner) + additional > self.config.max_storage_bytes:
            raise QuotaExceeded("Workspace storage quota exceeded.")

    def ingest(self, owner, path, chunks, *, text=False):
        """Bound bytes as they arrive; atomically replace only after full validation."""
        destination = self.path(owner, path)
        if text and destination.suffix.lower() not in {".xyz", ".cif", ".json"}:
            raise ValueError("Text uploads support .xyz, .cif and .json files.")
        limit = self.config.max_text_bytes if text else self.config.max_upload_bytes
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(prefix=".service-upload-", dir=destination.parent)
        size = 0
        try:
            with os.fdopen(fd, "wb") as stream:
                for chunk in chunks:
                    size += len(chunk)
                    if size > limit:
                        raise QuotaExceeded("Upload byte limit exceeded.")
                    # Include the immutable copy that will be published below.
                    self.check_quota(owner, len(chunk) + size)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
            self.path(owner, path)  # recheck before publication
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
        identifier = uuid.uuid4().hex
        context = SimpleNamespace(workspace=self.workspace(owner), task_id=identifier)
        manifests = output_manifest(context.workspace, [path])
        artifacts = capture_artifacts(context, identifier, manifests)
        references = self.publish(owner, identifier, context.workspace, artifacts)
        return {"path": path, **references[0]}

    def publish(self, owner, operation_id, workspace, descriptors):
        root = str(Path(workspace).relative_to(self.config.data_dir))
        refs = []
        with connection(self.config.database) as db:
            for artifact in descriptors:
                key = hashlib.sha256(f"{owner}:{operation_id}:{artifact['path']}".encode()).hexdigest()
                db.execute("INSERT OR IGNORE INTO artifacts VALUES (?, ?, ?, ?, ?)",
                           (key, owner, operation_id, root, json.dumps(artifact)))
                refs.append(self.reference(key, operation_id, artifact))
        return refs

    @staticmethod
    def reference(identifier, operation_id, artifact):
        name = artifact["source_path"]
        if name.startswith(".chemgraph-inputs/"):
            name = "inputs/" + Path(name).name
        return {"artifact_id": identifier, "reference": f"artifact:{identifier}",
                "operation_id": operation_id, "name": name,
                "size_bytes": artifact["size_bytes"], "sha256": artifact["sha256"],
                "media_type": artifact["media_type"], "download_path": f"/artifacts/{identifier}"}

    def list(self, owner, *, operation_id=None, limit=100, offset=0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Use a limit between 1 and 100 and a nonnegative offset.")
        query = "SELECT * FROM artifacts WHERE owner=?"
        params = [owner]
        if operation_id is not None:
            query += " AND operation_id=?"
            params.append(operation_id)
        with connection(self.config.database) as db:
            rows = db.execute(query + " ORDER BY rowid LIMIT ? OFFSET ?", [*params, limit, offset]).fetchall()
        return [self.reference(row["id"], row["operation_id"], json.loads(row["descriptor"])) for row in rows]

    def cataloged_operations(self):
        """Operations with catalog rows; publish() inserts each one's rows in one transaction."""
        with connection(self.config.database) as db:
            return {row[0] for row in db.execute("SELECT DISTINCT operation_id FROM artifacts")}

    def get(self, owner, identifier):
        with connection(self.config.database) as db:
            row = db.execute("SELECT * FROM artifacts WHERE owner=? AND id=?", (owner, identifier)).fetchone()
        if row is None:
            raise FileNotFoundError("Unknown artifact.")
        descriptor = json.loads(row["descriptor"])
        workspace = self.config.data_dir / row["workspace"]
        path = artifact_file(workspace, descriptor)
        verify_file(path, descriptor)
        return path, self.reference(row["id"], row["operation_id"], descriptor)

    def read(self, owner, identifier):
        path, reference = self.get(owner, identifier)
        if reference["size_bytes"] > self.config.max_text_bytes:
            raise QuotaExceeded("Artifact is too large for MCP text; use its authenticated download path.")
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raise ValueError("Binary artifact; use its authenticated download path.") from None
        return {**reference, "content": text}

    def source(self, owner, reference):
        if reference.startswith("artifact:"):
            return self.get(owner, reference.removeprefix("artifact:"))[0]
        return self.path(owner, reference)

    def stage(self, owner, source, destination):
        """Copy a validated source into a job, detecting mutation while copying."""
        from chemgraph.runtime.artifacts import publish_file

        path = self.source(owner, source)
        identity = file_identity(path)
        if identity["size_bytes"] > self.config.max_upload_bytes:
            raise QuotaExceeded("Calculation input exceeds the upload byte limit.")
        self.check_quota(owner, identity["size_bytes"] * 2)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        publish_file(destination.parent, path, destination, identity)
        return identity
