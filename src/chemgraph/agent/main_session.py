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
from chemgraph.memory.schemas import MainAgentGraphConfig, MainAgentSessionMetadata
from chemgraph.memory.serialization import serialize_messages
from chemgraph.memory.store import SessionStore


SessionStatus = Literal["completed", "waiting_for_user", "failed"]


class MainAgentRestoreError(RuntimeError):
    """Base error raised when a durable main-agent thread cannot be restored."""


class MissingCheckpointError(MainAgentRestoreError):
    """Raised when a stored session has no corresponding graph checkpoint."""


class IncompatibleCheckpointError(MainAgentRestoreError):
    """Raised when stored graph metadata is incompatible with the active graph."""


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
        session_store: SessionStore | None = None,
        session_metadata: MainAgentSessionMetadata | None = None,
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
        self.session_store = session_store
        self.session_metadata = session_metadata
        self._registered = False
        if self.session_store is not None:
            existing = self.session_store.get_session_metadata(self._thread_id)
            self._registered = existing is not None

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
        self._ensure_registered(message)
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

    async def restore(self) -> MainAgentTurnResult:
        """Restore pending, failed, or idle state without adding a message."""
        if self.session_store is not None:
            stored = self.session_store.get_session_metadata(self.thread_id)
            if stored is None:
                raise MainAgentRestoreError(
                    f"Session {self.thread_id!r} does not exist in the session store."
                )
            _, metadata = stored
            if metadata is not None and self.session_metadata is not None:
                stored_config = metadata.graph_config
                active_config = self.session_metadata.graph_config
                if stored_config.graph_schema_version != active_config.graph_schema_version:
                    raise IncompatibleCheckpointError(
                        "The stored graph schema is incompatible with this ChemGraph version."
                    )
                if (
                    stored_config.topology_fingerprint
                    and active_config.topology_fingerprint
                    and stored_config.topology_fingerprint
                    != active_config.topology_fingerprint
                ):
                    raise IncompatibleCheckpointError(
                        "The active graph topology does not match the stored session."
                    )

        snapshot = await self.workflow.aget_state(self.config)
        if not snapshot or not snapshot.created_at:
            raise MissingCheckpointError(
                f"No checkpoint exists for main-agent thread {self.thread_id!r}."
            )
        result = self._result_from_snapshot(snapshot)
        self._pending = result.interrupts
        self._failed = result.status == "failed"
        self._registered = True
        self._synchronize(snapshot.values, result.status)
        return result

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
        if self.session_store is not None:
            self.session_store.update_session_status(self.thread_id, "running")
        try:
            result = await self._run_once(stream_input)
        except Exception:
            self._failed = True
            try:
                snapshot = await self.workflow.aget_state(self.config)
                if snapshot and snapshot.values:
                    self._synchronize(snapshot.values, "failed")
                elif self.session_store is not None:
                    self.session_store.update_session_status(self.thread_id, "failed")
            except Exception:
                if self.session_store is not None:
                    self.session_store.update_session_status(self.thread_id, "failed")
            raise
        self._failed = False
        snapshot = await self.workflow.aget_state(self.config)
        self._synchronize(snapshot.values if snapshot else {}, result.status)
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

        snapshot = await self.workflow.aget_state(self.config)
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

    def _ensure_registered(self, message: str) -> None:
        if self.session_store is None or self._registered:
            return
        metadata = self.session_metadata or MainAgentSessionMetadata(
            graph_config=MainAgentGraphConfig(model_name="unknown")
        )
        self.session_store.create_session(
            session_id=self.thread_id,
            model_name=metadata.graph_config.model_name,
            workflow_type="main_agent",
            title=SessionStore.generate_title(message),
            status="new",
            session_metadata=metadata,
        )
        self._registered = True

    def _synchronize(
        self,
        state: Any,
        status: SessionStatus,
    ) -> None:
        if self.session_store is None or not self._registered:
            return
        values = state if isinstance(state, dict) else {}
        raw_messages = values.get("messages", [])
        self.session_store.synchronize_messages(
            self.thread_id,
            serialize_messages(list(raw_messages or [])),
        )
        self.session_store.update_session_status(self.thread_id, status)

    def _result_from_snapshot(self, snapshot: Any) -> MainAgentTurnResult:
        found = list(_pending_interrupts(getattr(snapshot, "interrupts", ())))
        failed = bool(getattr(snapshot, "next", ()))
        for task in snapshot.tasks:
            found.extend(_pending_interrupts(getattr(task, "interrupts", ())))
            failed = failed or bool(getattr(task, "error", None))
        pending = _deduplicate_interrupts(found)
        status: SessionStatus
        if pending:
            status = "waiting_for_user"
        elif failed:
            status = "failed"
        else:
            status = "completed"
        values = snapshot.values or {}
        return MainAgentTurnResult(
            thread_id=self.thread_id,
            status=status,
            assistant_response=latest_assistant_text(list(values.get("messages", []) or [])),
            interrupts=pending,
            state=serialize_state(values),
        )


__all__ = [
    "MainAgentSession",
    "MainAgentTurnResult",
    "MainAgentRestoreError",
    "MissingCheckpointError",
    "IncompatibleCheckpointError",
    "PendingInterrupt",
]
