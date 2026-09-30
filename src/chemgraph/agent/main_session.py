"""Session driver for the checkpointed ChemGraph supervisor graph."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from chemgraph.agent.events import EventCallback, _AstreamEventCallback
from chemgraph.agent.usage import UsageCollector, add_callbacks, session_usage
from chemgraph.agent.interrupts import (
    PendingInterrupt,
    collect_pending_interrupts,
    normalize_interrupts,
)
from chemgraph.agent.turn import serialize_state
from chemgraph.graphs.main_agent import latest_assistant_text, _preserve_terminal_tool_output
from chemgraph.memory.schemas import MainAgentGraphConfig, MainAgentSessionMetadata
from chemgraph.memory.graph_config import (
    GRAPH_SCHEMA_VERSION, NEW_SESSION_GUIDANCE, fingerprint, validate_configuration_id,
)
from chemgraph.memory.serialization import serialize_messages
from chemgraph.memory.store import SessionStore


logger = logging.getLogger(__name__)

SessionStatus = Literal["completed", "waiting_for_user", "failed"]


class MainAgentRestoreError(RuntimeError):
    """Base error raised when a durable main-agent thread cannot be restored."""


class MissingCheckpointError(MainAgentRestoreError):
    """Raised when a stored session has no corresponding graph checkpoint."""


class IncompatibleCheckpointError(MainAgentRestoreError):
    """Raised when stored graph metadata is incompatible with the active graph."""


@dataclass(frozen=True)
class MainAgentTurnResult:
    """Result returned when a supervisor turn completes or pauses."""

    thread_id: str
    status: SessionStatus
    assistant_response: str
    interrupts: tuple[PendingInterrupt, ...]
    state: dict[str, Any]
    usage: dict[str, Any] | None = None


class MainAgentSession:
    """Drive one checkpointed thread and validate its reconstruction identity.

    Pass ChemGraph's ``main_agent_metadata`` with its compiled workflow. Raw
    graphs use a non-secret ``configuration_id`` for reconstruction across
    session instances; without one only the current instance may continue.
    ``recursion_limit=None`` inherits metadata or the graph's configured limit.
    """

    def __init__(
        self,
        workflow: Any,
        *,
        thread_id: str | None = None,
        recursion_limit: int | None = None,
        session_store: SessionStore | None = None,
        session_metadata: MainAgentSessionMetadata | None = None,
        configuration_id: str | None = None,
        on_event: EventCallback | None = None,
    ):
        if recursion_limit is None:
            recursion_limit = (session_metadata.graph_config.recursion_limit if session_metadata
                               else (getattr(workflow, "config", None) or {}).get("recursion_limit", 200))
        if session_metadata is not None and recursion_limit != session_metadata.graph_config.recursion_limit:
            raise ValueError("recursion_limit must match the graph's session metadata.")
        if recursion_limit <= 0:
            raise ValueError("recursion_limit must be positive.")
        self.workflow = workflow
        configuration_id = validate_configuration_id(configuration_id)
        if session_metadata is not None:
            session_metadata = session_metadata.model_copy(deep=True)
            if configuration_id is not None and configuration_id != session_metadata.graph_config.configuration_id:
                raise ValueError("configuration_id must match the graph's session metadata.")
        else:
            session_metadata = MainAgentSessionMetadata(graph_config=MainAgentGraphConfig(
                model_name="unknown", graph_schema_version=GRAPH_SCHEMA_VERSION,
                configuration_id=configuration_id, requires_configuration_id=True,
                topology_fingerprint=(fingerprint({
                    "schema": GRAPH_SCHEMA_VERSION, "configuration_id": configuration_id,
                    "recursion_limit": recursion_limit,
                }) if configuration_id else ""),
            ))
        graph_config = session_metadata.graph_config
        self._owner_id = str(uuid.uuid4())
        self._restored = False
        self._thread_id = thread_id or str(uuid.uuid4())
        self.config = {
            "configurable": {"thread_id": self._thread_id},
            "recursion_limit": recursion_limit,
            "metadata": {
                "chemgraph_schema": graph_config.graph_schema_version,
                "chemgraph_topology": graph_config.topology_fingerprint,
                "chemgraph_owner": self._owner_id,
            },
        }
        if on_event is not None:
            self.config["callbacks"] = [
                _AstreamEventCallback(on_event, self._thread_id)
            ]
        self._failed = False
        self._pending: tuple[PendingInterrupt, ...] = ()
        self.session_store = session_store
        self.session_metadata = session_metadata
        self._usage: UsageCollector | None = None
        self._usage_turns: list[UsageCollector] = []
        self._usage_operation = 0
        self._history_unaccounted = False
        self._registered = False
        if self.session_store is not None:
            try:
                existing = self.session_store.get_session_metadata(self._thread_id)
            except Exception:
                logger.warning(
                    "Could not inspect readable storage for main-agent thread %s.",
                    self._thread_id,
                    exc_info=True,
                )
            else:
                self._registered = existing is not None

    @property
    def thread_id(self) -> str:
        """Return the stable LangGraph thread identifier."""
        return self._thread_id

    @property
    def last_usage(self) -> dict | None:
        """Provider usage for the latest query, including retries and workers."""
        return self._usage.summary if self._usage is not None else None

    @property
    def session_usage(self) -> dict:
        """All recorded turns in this session, including restored history."""
        return session_usage(
            self._usage_turns, self.session_store, self.thread_id,
            history_unaccounted=self._history_unaccounted,
        )

    def _start_usage(self, restored: dict | None = None) -> None:
        metadata = self.session_metadata
        self._usage = UsageCollector(
            self.thread_id, self.thread_id, store=self.session_store,
            model=metadata.graph_config.model_name if metadata else None,
            turn_id=restored["turn_id"] if restored else None,
            records=restored["records"] if restored else (),
        )
        self._usage_turns.append(self._usage)

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
        snapshot = await self._validate_checkpoint()
        if snapshot and getattr(snapshot, "created_at", None):
            if not self._restored and (snapshot.metadata or {}).get("chemgraph_owner") != self._owner_id:
                raise RuntimeError("Restore the existing session before starting a new turn.")
            status = self._result_from_snapshot(snapshot).status
            if status != "completed":
                raise RuntimeError("Restore the existing session before resuming or retrying it.")
        self._ensure_registered(message)
        self._start_usage()
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
        await self._validate_checkpoint()
        return await self._run(Command(resume=self._resume_value(response)))

    async def retry(self) -> MainAgentTurnResult:
        """Resume the failed checkpoint without duplicating user input."""
        if not self._failed:
            raise RuntimeError("The main-agent session has no failed operation to retry.")
        await self._validate_checkpoint()
        return await self._run(None)

    async def restore(self) -> MainAgentTurnResult:
        """Restore pending, failed, or idle state without adding a message."""
        snapshot = await self._validate_checkpoint()
        if not snapshot or not snapshot.created_at:
            raise MissingCheckpointError(
                f"No checkpoint exists for main-agent thread {self.thread_id!r}."
            )
        result = self._result_from_snapshot(snapshot)
        self._restored = True
        self._pending = result.interrupts
        self._failed = result.status == "failed"
        self._registered = True
        has_history = bool((snapshot.values or {}).get("messages"))
        restored = None
        if self.session_store is not None:
            try:
                restored = self.session_store.latest_usage_turn(self.thread_id, self.thread_id)
                if restored is not None:
                    # Preserve any in-memory calls whose persistence failed.
                    if self._usage is None or self._usage.turn_id != restored["turn_id"]:
                        self._start_usage(restored)
                self._history_unaccounted |= self.session_store.usage_history_unaccounted(self.thread_id)
            except Exception:
                logger.warning("Could not restore token usage.", exc_info=True)
        if has_history and restored is None and not self._usage_turns:
            self._history_unaccounted = True
        if self._history_unaccounted and self.session_store is not None:
            try:
                self.session_store.mark_usage_history_unaccounted(self.thread_id)
            except Exception:
                logger.warning("Could not persist historical usage coverage.", exc_info=True)
        self._synchronize(snapshot.values, result.status)
        return replace(result, usage=self.last_usage)

    async def _validate_checkpoint(self):
        """Check both stores before any existing thread can execute or be resumed."""
        active = self.session_metadata.graph_config
        if active.graph_schema_version != GRAPH_SCHEMA_VERSION:
            raise IncompatibleCheckpointError(f"The active graph schema is incompatible. {NEW_SESSION_GUIDANCE}")
        snapshot = await self.workflow.aget_state(self.config)
        checkpoint = (snapshot.metadata or {}) if getattr(snapshot, "created_at", None) else None
        same_owner = checkpoint is not None and checkpoint.get("chemgraph_owner") == self._owner_id

        def validate(schema, topology):
            if schema != GRAPH_SCHEMA_VERSION:
                raise IncompatibleCheckpointError(
                    f"The stored graph schema is incompatible. {NEW_SESSION_GUIDANCE}"
                )
            if not same_owner and active.requires_configuration_id and not active.configuration_id:
                raise IncompatibleCheckpointError(
                    "Restoring caller-owned configuration requires the original configuration_id."
                )
            if not topology or not active.topology_fingerprint:
                if same_owner and not topology and not active.topology_fingerprint:
                    return
                raise IncompatibleCheckpointError(
                    f"The stored or active graph has no topology identity. {NEW_SESSION_GUIDANCE}"
                )
            if topology != active.topology_fingerprint:
                raise IncompatibleCheckpointError(
                    "The active graph topology does not match the stored session. "
                    + NEW_SESSION_GUIDANCE
                )

        if checkpoint is not None:
            validate(checkpoint.get("chemgraph_schema"), checkpoint.get("chemgraph_topology"))
        if self.session_store is not None:
            try:
                stored = self.session_store.get_session_metadata(self.thread_id)
            except Exception:
                # Readable storage is supplementary; checkpoints remain authoritative.
                logger.warning("Could not inspect readable session metadata.", exc_info=True)
            else:
                if stored is not None:
                    metadata = stored[1]
                    validate(
                        metadata.graph_config.graph_schema_version if metadata else None,
                        metadata.graph_config.topology_fingerprint if metadata else None,
                    )
        return snapshot

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
        if self._usage is None:
            self._start_usage()
        self._usage_operation += 1
        self._update_status("running")
        try:
            result, state_values = await self._run_once(stream_input)
        except Exception:
            self._usage.finish("failed")
            self._failed = True
            try:
                snapshot = await self.workflow.aget_state(self.config)
            except Exception:
                snapshot = None
                logger.debug(
                    "Could not inspect checkpoint after main-agent failure.",
                    exc_info=True,
                )
            if snapshot and snapshot.values:
                self._synchronize(snapshot.values, "failed")
            else:
                self._update_status("failed")
            raise
        except BaseException:
            self._usage.finish("cancelled")
            raise
        self._failed = False
        self._synchronize(state_values, result.status)
        self._usage.finish(result.status)
        return replace(result, usage=self.last_usage)

    async def _run_once(
        self, stream_input: Any
    ) -> tuple[MainAgentTurnResult, dict[str, Any]]:
        last_state: dict[str, Any] | None = None
        found: list[PendingInterrupt] = []
        try:
            async for state in self.workflow.astream(
                stream_input,
                stream_mode="values",
                config=add_callbacks(self.config, [self._usage]) if self._usage else self.config,
            ):
                last_state = state
                found.extend(normalize_interrupts(state.get("__interrupt__")))
        except GraphInterrupt as exc:
            raw_interrupts = exc.args[0] if exc.args else []
            found.extend(normalize_interrupts(raw_interrupts))

        snapshot = await self.workflow.aget_state(self.config)
        state_values = snapshot.values if snapshot else (last_state or {})
        pending = collect_pending_interrupts(found, snapshot)
        self._pending = pending
        result = MainAgentTurnResult(
            thread_id=self.thread_id,
            status="waiting_for_user" if pending else "completed",
            assistant_response=latest_assistant_text(
                list((_preserve_terminal_tool_output(state_values) if not pending
                      else state_values).get("messages", []) or [])
            ),
            interrupts=pending,
            state=serialize_state(state_values),
        )
        return result, state_values

    def _ensure_registered(self, message: str) -> None:
        if self.session_store is None or self._registered:
            return
        metadata = self.session_metadata
        try:
            self.session_store.create_session(
                session_id=self.thread_id,
                model_name=metadata.graph_config.model_name,
                workflow_type="main_agent",
                title=SessionStore.generate_title(message),
                status="new",
                session_metadata=metadata,
            )
        except Exception:
            logger.warning(
                "Could not register readable storage for main-agent thread %s.",
                self.thread_id,
                exc_info=True,
            )
            try:
                self._registered = (
                    self.session_store.get_session_metadata(self.thread_id) is not None
                )
            except Exception:
                pass
        else:
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
        try:
            self.session_store.synchronize_messages(
                self.thread_id,
                serialize_messages(list(raw_messages or [])),
            )
        except Exception:
            logger.warning(
                "Could not synchronize the readable transcript for main-agent "
                "thread %s.",
                self.thread_id,
                exc_info=True,
            )
        self._update_status(status)

    def _update_status(self, status: str) -> None:
        if self.session_store is None or not self._registered:
            return
        try:
            self.session_store.update_session_status(self.thread_id, status)
        except Exception:
            logger.warning(
                "Could not update readable status for main-agent thread %s.",
                self.thread_id,
                exc_info=True,
            )

    def _result_from_snapshot(self, snapshot: Any) -> MainAgentTurnResult:
        failed = bool(getattr(snapshot, "next", ()))
        for task in snapshot.tasks:
            failed = failed or bool(getattr(task, "error", None))
        pending = collect_pending_interrupts([], snapshot)
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
            assistant_response=latest_assistant_text(list(
                (_preserve_terminal_tool_output(values) if status == "completed"
                 else values).get("messages", []) or []
            )),
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
