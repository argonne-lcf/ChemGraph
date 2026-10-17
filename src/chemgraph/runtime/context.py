"""Caller identity, workspace confinement, and output registration for scoped runs.

This is an application boundary, not an operating-system sandbox. Calculator
dependencies and model files remain operator-provisioned resources.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from chemgraph.runtime.artifacts import INPUTS_DIR, INTERNAL_DIRS, checked_path

if TYPE_CHECKING:
    from chemgraph.runtime.journal import OperationJournal
    from chemgraph.runtime.policy import ExecutionPolicy, PreparedAction


class ScopedExecutionError(RuntimeError):
    """Execution control failed; the run must stop rather than retry the action."""


@dataclass(frozen=True)
class WorkspaceScope:
    """The immutable, serializable filesystem boundary of an invocation."""

    owner: str
    workspace: Path

    def __post_init__(self):
        if not self.owner:
            raise ValueError("A workspace requires an owner.")
        object.__setattr__(self, "workspace", Path(self.workspace).resolve())


@dataclass
class RunContext:
    """Authenticated caller scope for one conversation.

    Keep one instance per conversation and change only ``task_id`` between
    serialized turns. ``workspace`` must be distinct per conversation.
    """

    owner: str
    thread_id: str
    task_id: str
    workspace: Path
    policy: ExecutionPolicy
    journal: OperationJournal
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self):
        if not all((self.owner, self.thread_id, self.task_id)):
            raise ValueError("A scoped run requires owner, thread and task identities.")
        self.workspace = Path(self.workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    @property
    def scope(self) -> WorkspaceScope:
        # Derive this when binding so existing workspace relocation remains valid.
        return WorkspaceScope(self.owner, self.workspace)


_context: ContextVar[RunContext | None] = ContextVar("chemgraph_run_context", default=None)
_workspace: ContextVar[WorkspaceScope | None] = ContextVar("chemgraph_workspace", default=None)
_inputs: ContextVar[frozenset[str]] = ContextVar("chemgraph_inputs", default=frozenset())
_outputs: ContextVar[set[str] | None] = ContextVar("chemgraph_outputs", default=None)


def current_context() -> RunContext | None:
    return _context.get()


def current_workspace() -> WorkspaceScope | None:
    return _workspace.get()


@contextmanager
def workspace_scope(scope: WorkspaceScope, *, inputs=(), outputs=None):
    """Bind only worker-safe data, with no journal, policy, or event-loop objects."""
    workspace = _workspace.set(scope)
    permitted = _inputs.set(frozenset(inputs))
    registered = _outputs.set(outputs)
    try:
        yield scope
    finally:
        _outputs.reset(registered)
        _inputs.reset(permitted)
        _workspace.reset(workspace)


@contextmanager
def run_context(context: RunContext):
    """Bind ``context`` around an asynchronous session run, resume or retry."""
    token = _context.set(context)
    try:
        with workspace_scope(context.scope):
            yield context
    finally:
        _context.reset(token)


@contextmanager
def invocation(action: PreparedAction, outputs: set[str]):
    """Expose the approved preparation and collect registered outputs."""
    preparation = _inputs.set(frozenset(item["path"] for item in action.inputs))
    registered = _outputs.set(outputs)
    try:
        yield
    finally:
        _outputs.reset(registered)
        _inputs.reset(preparation)


def scoped_path(path) -> str | None:
    """Resolve a tool path inside the active workspace; ``None`` when unscoped.

    Rejects traversal and symlink escapes. Reserved directories are not
    tool-controlled, except the input snapshots of the approved preparation.
    """
    context = current_workspace()
    if context is None:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = context.workspace / candidate
    if INTERNAL_DIRS.intersection(candidate.parts):
        checked_path(context.workspace, candidate)
    resolved = candidate.resolve()
    if not resolved.is_relative_to(context.workspace):
        raise PermissionError("Tool path is outside this conversation's workspace.")
    relative = resolved.relative_to(context.workspace)
    if INTERNAL_DIRS.intersection(relative.parts):
        if relative.parts[0] != INPUTS_DIR or relative.as_posix() not in _inputs.get():
            raise PermissionError("Internal execution records are not tool-controlled paths.")
    return str(resolved)


def register_artifact(path) -> None:
    """Register a written output of the current scoped invocation.

    Call after writing the file. Outside a scoped invocation this is a no-op,
    so tools keep their existing result formats.
    """
    outputs, context = _outputs.get(), current_workspace()
    if outputs is not None and context is not None:
        resolved = Path(scoped_path(path))
        outputs.add(resolved.relative_to(context.workspace).as_posix())


def standalone_log_record(record) -> bool:
    """Logging filter: process-wide log files never collect a scoped caller's logs."""
    return current_workspace() is None
