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


def write_text(path, text):
    """Durably replace a file so readers see the old or new content, never a mix."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".hpc-write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


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
def locked_run(directory, *, roots, create=False):
    """Lock a run beneath a configured root; only staging may create it.

    Relative paths are rejected: the per-session artifact directory changes
    across restarts, which would detach a run from its saved location.
    """
    # Imported here so non-POSIX core installations can still discover tools.
    try:
        import fcntl
    except ImportError:
        raise RuntimeError("HPC run locking requires a POSIX agent host.") from None

    path = Path(directory)
    if not path.is_absolute():
        raise ValueError("Use an absolute host run directory beneath a target local_root.")
    path = path.resolve()
    if not any(path != root and path.is_relative_to(root) for root in roots):
        raise ValueError("Run directory must be beneath a configured target local_root.")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise ValueError(f"Run directory does not exist: {path}")
    fd = os.open(path / ".hpc.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield path
    finally:
        os.close(fd)


def mark_started(path, attempt_id):
    """Record that ``attempt_id`` may have been sent; it must never be sent again."""
    write_text(path / "submission.started", attempt_id)


def started_attempt(path):
    marker = path / "submission.started"
    return marker.read_text().strip() if marker.exists() else None
