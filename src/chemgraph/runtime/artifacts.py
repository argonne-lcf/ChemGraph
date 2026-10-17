"""Checked internal paths and durable, bounded-memory file publication.

Persisted descriptors use workspace-relative paths so a conversation workspace
restored to a different location still verifies and replays.
"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import tempfile
from pathlib import Path

INPUTS_DIR = ".chemgraph-inputs"
ARTIFACTS_DIR = ".chemgraph-artifacts"
INTERNAL_DIRS = frozenset({INPUTS_DIR, ARTIFACTS_DIR})

_CHUNK = 1024 * 1024


def checked_path(root, path) -> Path:
    """Return ``path`` if it lies under ``root`` without symlinked components."""
    root, path = Path(root), Path(path)
    if not path.is_relative_to(root) or ".." in path.parts:
        raise ValueError(f"Invalid execution record path: {path}")
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        if candidate.is_symlink():
            raise ValueError(f"Execution records cannot be symlinks: {path}")
    return path


def internal_path(workspace, directory: str, *parts) -> Path:
    """Resolve a path inside one of the reserved workspace directories."""
    root = Path(workspace) / directory
    path = checked_path(workspace, root.joinpath(*parts))
    if not path.is_relative_to(root):
        raise ValueError(f"Invalid execution record path: {path}")
    return path


def sync_directory(path) -> None:
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def file_identity(path) -> dict:
    """Return the SHA-256 digest and size of a regular file, read in chunks."""
    if not Path(path).is_file():
        raise ValueError(f"Execution file is missing or not a regular file: {path}")
    checksum, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(_CHUNK):
            checksum.update(chunk)
            size += len(chunk)
    return {"sha256": checksum.hexdigest(), "size_bytes": size}


def verify_file(path, expected: dict) -> None:
    identity = file_identity(path)
    if identity != {key: expected[key] for key in ("sha256", "size_bytes")}:
        raise ValueError(f"Execution file changed: {path}")


def publish_file(workspace, source, destination, expected: dict) -> None:
    """Publish exact bytes atomically, or verify and sync an earlier publication."""
    workspace = Path(workspace)
    destination = checked_path(workspace, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        verify_file(destination, expected)
        # A previous attempt may have failed after rename but before syncing.
        # Windows flushes only handles with write access; "r+b" never truncates.
        with destination.open("r+b") as stream:
            os.fsync(stream.fileno())
    else:
        fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=destination.parent)
        try:
            checksum, size = hashlib.sha256(), 0
            with os.fdopen(fd, "wb") as target, Path(source).open("rb") as stream:
                while chunk := stream.read(_CHUNK):
                    target.write(chunk)
                    checksum.update(chunk)
                    size += len(chunk)
                if checksum.hexdigest() != expected["sha256"] or size != expected["size_bytes"]:
                    raise ValueError(f"Execution file changed: {source}")
                target.flush()
                os.fsync(target.fileno())
            checked_path(workspace, destination)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)
    # Include newly created ancestor entries, up to the existing workspace.
    for directory in destination.parents:
        sync_directory(directory)
        if directory == workspace:
            break


def output_manifest(workspace, outputs) -> list[dict]:
    """Identify registered outputs (workspace-relative paths) by content."""
    manifest = []
    for relative in sorted(outputs):
        source = checked_path(workspace, Path(workspace) / relative)
        manifest.append({"source_path": relative, **file_identity(source)})
    return manifest


def capture_artifacts(context, tool_call_id: str, manifest: list[dict]) -> list[dict]:
    """Publish immutable copies of exactly the outputs recorded in ``manifest``."""
    artifacts = []
    for output in manifest:
        relative, digest = output["source_path"], output["sha256"]
        source = checked_path(context.workspace, context.workspace / relative)
        key = hashlib.sha256(
            f"{context.task_id}:{tool_call_id}:{relative}:{digest}".encode()
        ).hexdigest()
        destination = internal_path(context.workspace, ARTIFACTS_DIR, key, source.name)
        publish_file(context.workspace, source, destination, output)
        artifacts.append({
            "path": destination.relative_to(context.workspace).as_posix(),
            "media_type": mimetypes.guess_type(source.name)[0] or "application/octet-stream",
            "task_id": context.task_id,
            "tool_call_id": tool_call_id,
            "source_path": relative,
            "sha256": digest,
            "size_bytes": output["size_bytes"],
        })
    return artifacts


def artifact_file(workspace, artifact: dict) -> Path:
    """Return the published file for a descriptor, rejecting paths outside the store."""
    return internal_path(workspace, ARTIFACTS_DIR, Path(artifact["path"]).relative_to(ARTIFACTS_DIR))


def verify_artifacts(workspace, artifacts: list[dict]) -> None:
    for artifact in artifacts:
        verify_file(artifact_file(workspace, artifact), artifact)
