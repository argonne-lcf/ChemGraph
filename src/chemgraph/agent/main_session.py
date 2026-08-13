"""Session driver for the checkpointed ChemGraph supervisor graph."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from chemgraph.agent.turn import serialize_state
from chemgraph.graphs.main_agent import latest_assistant_text


SessionStatus = Literal["completed", "waiting_for_user"]


@dataclass(frozen=True)
class PendingInterrupt:
    """One pending request for user input."""

    id: str
    payload: Any


@dataclass(frozen=True)
class MainAgentTurnResult:
    """Result returned when a supervisor turn completes or pauses."""

    thread_id: str
    status: SessionStatus
    assistant_response: str
    interrupts: tuple[PendingInterrupt, ...]
    state: dict[str, Any]


def _pending_interrupts(values: Any) -> list[PendingInterrupt]:
    if values is None:
        return []
    if not isinstance(values, (list, tuple)):
        values = [values]

    return [
        PendingInterrupt(
            id=str(getattr(item, "id", "") or ""),
            payload=getattr(item, "value", item),
        )
        for item in values
    ]


def _deduplicate_interrupts(
    interrupts: list[PendingInterrupt],
) -> tuple[PendingInterrupt, ...]:
    unique: list[PendingInterrupt] = []
    seen: set[str] = set()
    for item in interrupts:
        key = item.id or repr(item.payload)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return tuple(unique)


class MainAgentSession:
    """Drive one resumable, process-lifetime main-agent thread."""

    def __init__(
        self,
        workflow: Any,
        *,
        thread_id: str | None = None,
        recursion_limit: int = 50,
    ):
        if recursion_limit <= 0:
            raise ValueError("recursion_limit must be positive.")
        self.workflow = workflow
        self._thread_id = thread_id or str(uuid.uuid4())
        self.config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": recursion_limit,
        }
        self._failed = False
        self._pending: tuple[PendingInterrupt, ...] = ()

    @property
    def thread_id(self) -> str:
        """Return the stable LangGraph thread identifier."""
        return self._thread_id

    @property
    def pending_interrupts(self) -> tuple[PendingInterrupt, ...]:
        """Return the current pending user-input requests."""
        return self._pending

    @property
    def failed(self) -> bool:
        """Return whether the most recent graph operation raised an error."""
        return self._failed

    async def run(self, message: str) -> MainAgentTurnResult:
        """Run a normal user turn on this checkpointed thread."""
        if self._failed:
            raise RuntimeError(
                "The main-agent session failed; retry it before running a new turn."
            )
        if self._pending:
            raise RuntimeError(
                "The main-agent session is waiting for interrupt responses."
            )
        if not isinstance(message, str) or not message.strip():
            raise ValueError("The user message must be a non-empty string.")
        return await self._run({"messages": [HumanMessage(content=message)]})

    async def resume(
        self,
        response: str | Mapping[str, Any],
    ) -> MainAgentTurnResult:
        """Answer one or more pending nested-graph interrupts."""
        if self._failed:
            raise RuntimeError(
                "The main-agent session failed; retry it before resuming."
            )
        if not self._pending:
            raise RuntimeError("The main-agent session is not waiting for input.")
        return await self._run(Command(resume=self._resume_value(response)))

    async def retry(self) -> MainAgentTurnResult:
        """Resume the failed checkpoint without duplicating user input."""
        if not self._failed:
            raise RuntimeError("The main-agent session has no failed operation to retry.")
        return await self._run(None)

    def _resume_value(self, response: str | Mapping[str, Any]) -> Any:
        if isinstance(response, str):
            if len(self._pending) != 1:
                raise ValueError(
                    "Multiple interrupts require a mapping from interrupt ID "
                    "to response."
                )
            if not response.strip():
                raise ValueError("The interrupt response must not be empty.")
            return response

        expected = {item.id for item in self._pending}
        if "" in expected:
            raise RuntimeError("Pending interrupts do not expose stable IDs.")
        provided = {str(key) for key in response}
        if len(self._pending) == 1 and provided != expected:
            return dict(response)
        if provided != expected:
            raise ValueError(
                "Interrupt response IDs must exactly match the pending interrupts."
            )
        return {str(key): value for key, value in response.items()}

    async def _run(self, stream_input: Any) -> MainAgentTurnResult:
        try:
            result = await self._run_once(stream_input)
        except Exception:
            self._failed = True
            raise
        self._failed = False
        return result

    async def _run_once(self, stream_input: Any) -> MainAgentTurnResult:
        last_state: dict[str, Any] | None = None
        found: list[PendingInterrupt] = []
        try:
            async for state in self.workflow.astream(
                stream_input,
                stream_mode="values",
                config=self.config,
            ):
                last_state = state
                found.extend(_pending_interrupts(state.get("__interrupt__")))
        except GraphInterrupt as exc:
            raw_interrupts = exc.args[0] if exc.args else []
            found.extend(_pending_interrupts(raw_interrupts))

        snapshot = self.workflow.get_state(self.config)
        state_values = snapshot.values if snapshot else (last_state or {})
        if snapshot:
            for task in snapshot.tasks:
                found.extend(_pending_interrupts(getattr(task, "interrupts", ())))

        pending = _deduplicate_interrupts(found)
        self._pending = pending
        return MainAgentTurnResult(
            thread_id=self.thread_id,
            status="waiting_for_user" if pending else "completed",
            assistant_response=latest_assistant_text(
                list(state_values.get("messages", []) or [])
            ),
            interrupts=pending,
            state=serialize_state(state_values),
        )


__all__ = [
    "MainAgentSession",
    "MainAgentTurnResult",
    "PendingInterrupt",
]
