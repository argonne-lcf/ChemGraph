"""Caller-scoped, reviewed and durable tool execution for embedding services.

A service builds a main agent with ``approval_mode="scoped"`` and runs each
turn inside :func:`run_context`. Every permitted tool call is prepared exactly
(:mod:`chemgraph.runtime.policy`), paused for a decision recorded in the
:class:`OperationJournal`, executed once with its prepared arguments, and its
registered outputs are published as immutable artifacts. This is an
application boundary, not an operating-system sandbox.

:class:`ScopedToolsMiddleware` lives in :mod:`chemgraph.runtime.middleware`
so tool modules can import this package without LangChain agent machinery.
"""

from chemgraph.runtime.context import (
    RunContext,
    WorkspaceScope,
    ScopedExecutionError,
    current_context,
    current_workspace,
    register_artifact,
    run_context,
    scoped_path,
    workspace_scope,
)
from chemgraph.runtime.journal import (
    ApprovalRequired,
    FileOperationStore,
    OperationConflict,
    OperationJournal,
    OperationStore,
    result_artifacts,
)
from chemgraph.runtime.artifacts import artifact_file
from chemgraph.runtime.policy import (
    ExecutionPolicy,
    PolicyDenied,
    PreparationError,
    PreparedAction,
)

__all__ = [
    "ApprovalRequired",
    "ExecutionPolicy",
    "FileOperationStore",
    "OperationConflict",
    "OperationJournal",
    "OperationStore",
    "PolicyDenied",
    "PreparationError",
    "PreparedAction",
    "RunContext",
    "WorkspaceScope",
    "ScopedExecutionError",
    "artifact_file",
    "current_context",
    "current_workspace",
    "register_artifact",
    "result_artifacts",
    "run_context",
    "scoped_path",
    "workspace_scope",
]
