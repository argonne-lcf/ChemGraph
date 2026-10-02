"""Artifact directories scoped to one durable session operation."""

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path


_artifact_directory: ContextVar[str | None] = ContextVar("artifact_directory", default=None)


def canonical_artifact_directory(path) -> str:
    """Freeze a caller's directory before cwd or environment can change."""
    return str(Path(path).expanduser().resolve())


def artifact_directory() -> str | None:
    """Prefer the active session, retaining the standalone environment contract."""
    return _artifact_directory.get() or os.environ.get("CHEMGRAPH_LOG_DIR")


@contextmanager
def artifact_context(directory: str | None):
    """Propagate through async tasks and context-aware tool executors."""
    token = _artifact_directory.set(directory)
    try:
        yield
    finally:
        _artifact_directory.reset(token)
