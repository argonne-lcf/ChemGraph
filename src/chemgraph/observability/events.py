from __future__ import annotations

import contextlib
import contextvars
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from chemgraph.academy.observability.event_log import EventLog


def new_span_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


@dataclass(frozen=True)
class WorkflowEventContext:
    """Execution context for nested ChemGraph workflow events."""

    run_id: str | None
    run_dir: str | None
    agent_id: str | None
    role: str | None
    parent_span_id: str | None
    tool_name: str | None
    runtime: str = "chemgraph-langgraph"


@dataclass(frozen=True)
class WorkflowEventSink:
    """Write normalized workflow events to canonical Academy events."""

    path: Path
    context: WorkflowEventContext

    def emit(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        *,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        runtime: str | None = None,
        agent_id: str | None = None,
        role: str | None = None,
    ) -> dict[str, Any]:
        ctx = self.context
        resolved_agent_id = agent_id or ctx.agent_id
        resolved_role = role or ctx.role
        body = {
            **(payload or {}),
            "span_id": span_id,
            "parent_span_id": parent_span_id or ctx.parent_span_id,
            "runtime": runtime or ctx.runtime,
            "run_id": ctx.run_id,
            "run_dir": ctx.run_dir,
            "agent_id": resolved_agent_id,
            "role": resolved_role,
            "parent_tool_name": ctx.tool_name,
            "nested": True,
        }
        record = EventLog(self.path).emit(
            event,  # type: ignore[arg-type]
            run_id=ctx.run_id,
            agent_id=resolved_agent_id or "system",
            role=resolved_role,
            correlation_id=span_id,
            payload=body,
        )
        return record.model_dump(mode="json")


_CURRENT_SINK: contextvars.ContextVar[WorkflowEventSink | None] = (
    contextvars.ContextVar("chemgraph_workflow_event_sink", default=None)
)
_CURRENT_CONTEXT: contextvars.ContextVar[WorkflowEventContext | None] = (
    contextvars.ContextVar("chemgraph_workflow_event_context", default=None)
)


def current_workflow_event_context() -> WorkflowEventContext | None:
    return _CURRENT_CONTEXT.get()


def emit_workflow_event(
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    span_id: str | None = None,
    parent_span_id: str | None = None,
    runtime: str | None = None,
) -> dict[str, Any] | None:
    sink = _CURRENT_SINK.get()
    if sink is None:
        return None
    return sink.emit(
        event,
        payload,
        span_id=span_id,
        parent_span_id=parent_span_id,
        runtime=runtime,
    )


@contextlib.contextmanager
def workflow_event_context(
    *,
    jsonl_path: str | Path,
    context: WorkflowEventContext,
) -> Iterator[WorkflowEventSink]:
    sink = WorkflowEventSink(Path(jsonl_path), context=context)
    sink_token = _CURRENT_SINK.set(sink)
    context_token = _CURRENT_CONTEXT.set(context)
    try:
        yield sink
    finally:
        _CURRENT_CONTEXT.reset(context_token)
        _CURRENT_SINK.reset(sink_token)
