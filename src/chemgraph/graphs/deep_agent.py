"""Reusable Deep Agent workflow for workspace and coding tasks."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol
from langgraph.checkpoint.memory import InMemorySaver


DEFAULT_DEEPAGENT_PROMPT = """\
You are ChemGraph's workspace specialist. Complete repository exploration,
coding, testing, file analysis, and other multi-step workspace tasks. Use the
built-in filesystem and execution tools when needed. Treat `/workspace` as the
project root when that virtual mount is available. The execution tool runs in a
host shell, so follow any "Shell paths vs. virtual paths" mapping in the system
instructions when passing file paths to shell commands. Do not perform
molecular simulations or invent chemistry results; return those tasks to the
supervisor for delegation to the `chemgraph` specialist.

The calling supervisor sees only your final assistant message. Return a concise,
self-contained report including important results, changed paths, commands run,
and any failures or unresolved risks.
"""


DEFAULT_DEEPAGENT_INTERRUPT_ON = {
    "execute": {"allowed_decisions": ["approve", "reject"]},
    "write_file": {"allowed_decisions": ["approve", "reject"]},
    "edit_file": {"allowed_decisions": ["approve", "reject"]},
    "delete": {"allowed_decisions": ["approve", "reject"]},
}


_DEFAULT_CHECKPOINTER = object()
_DEFAULT_INTERRUPT_POLICY = object()
_WORKSPACE_MOUNT = "/workspace/"


def _normalize_backend(backend: BackendProtocol) -> BackendProtocol:
    """Mount a virtual local workspace at the path Deep Agent expects."""
    if isinstance(backend, LocalShellBackend) and backend.virtual_mode:
        return CompositeBackend(
            default=backend,
            routes={_WORKSPACE_MOUNT: backend},
        )
    return backend


def construct_deep_agent_graph(
    llm: Any,
    *,
    tools: Sequence[Any] | None = None,
    system_prompt: str = DEFAULT_DEEPAGENT_PROMPT,
    backend: BackendProtocol | None = None,
    interrupt_on: dict[str, Any] | None | object = _DEFAULT_INTERRUPT_POLICY,
    recursion_limit: int = 50,
    checkpointer: Any = _DEFAULT_CHECKPOINTER,
    name: str = "deepagent",
):
    """Construct a standalone or parent-checkpointed workspace Deep Agent.

    Standalone construction receives an in-memory checkpointer so approval
    interrupts can be resumed. Orchestrators should explicitly pass
    ``checkpointer=None`` so the worker inherits the parent graph's checkpoint.
    Passing ``interrupt_on=None`` disables approval interrupts and should be
    reserved for an externally isolated, explicitly trusted execution context.
    """
    if recursion_limit <= 0:
        raise ValueError("recursion_limit must be positive.")

    effective_checkpointer = (
        InMemorySaver()
        if checkpointer is _DEFAULT_CHECKPOINTER
        else checkpointer
    )
    effective_interrupt_on = (
        deepcopy(DEFAULT_DEEPAGENT_INTERRUPT_ON)
        if interrupt_on is _DEFAULT_INTERRUPT_POLICY
        else interrupt_on
    )
    effective_backend = _normalize_backend(
        backend if backend is not None else StateBackend()
    )
    workflow = create_deep_agent(
        model=llm,
        tools=list(tools or []),
        system_prompt=system_prompt,
        backend=effective_backend,
        interrupt_on=effective_interrupt_on,
        checkpointer=effective_checkpointer,
        name=name,
    )
    return workflow.with_config({"recursion_limit": recursion_limit})


__all__ = [
    "DEFAULT_DEEPAGENT_INTERRUPT_ON",
    "DEFAULT_DEEPAGENT_PROMPT",
    "construct_deep_agent_graph",
]
