"""Per-run evidence with process locks and durable atomic writes."""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".hpc-write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    return json.loads(Path(path).read_text())


def checksum(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def locked_run(directory):
    # Imported here so non-POSIX core installations can still discover tools.
    try:
        import fcntl
    except ImportError:
        raise RuntimeError("HPC run locking requires a POSIX agent host.") from None
    from chemgraph.tools.ase_core import _resolve_path

    path = Path(_resolve_path(directory)).resolve()
    path.mkdir(parents=True, exist_ok=True)
    fd = os.open(path / ".hpc.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)


def mark_started(path):
    fd = os.open(
        path / "submission.started", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    with os.fdopen(fd, "w") as stream:
        stream.write(now())
        stream.flush()
        os.fsync(stream.fileno())
    sync_directory(path)
