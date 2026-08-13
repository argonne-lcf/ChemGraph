"""Persistence adapter for direct main-agent subgraph invocations."""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.errors import GraphInterrupt

from chemgraph.memory.serialization import serialize_messages
from chemgraph.memory.store import SessionStore


_RUN_NAMESPACE = uuid.UUID("7ceaa8ee-c7ac-4efd-99e6-b8e3ad89de82")


class SubagentRunRecorder:
    """Record direct subagent lifecycle events in a :class:`SessionStore`."""

    def __init__(self, store: SessionStore):
        self.store = store

    @staticmethod
    def _identity(agent_name: str, state: Any, config: dict) -> tuple[str, str, str]:
        configurable = config.get("configurable", {})
        thread_id = str(configurable.get("thread_id", ""))
        checkpoint_namespace = str(configurable.get("checkpoint_ns", ""))
        if not thread_id or not checkpoint_namespace:
            raise RuntimeError("Subagent recording requires thread and checkpoint IDs.")
        run_id = str(
            uuid.uuid5(
                _RUN_NAMESPACE,
                f"{thread_id}\0{checkpoint_namespace}\0{agent_name}",
            )
        )
        messages = state.get("messages", []) if isinstance(state, dict) else []
        task = str(getattr(messages[-1], "content", "")) if messages else ""
        return run_id, thread_id, task

    def start(self, agent_name: str, state: Any, config: dict) -> str:
        run_id, thread_id, task = self._identity(agent_name, state, config)
        checkpoint_namespace = str(config["configurable"]["checkpoint_ns"])
        self.store.upsert_subagent_run(
            run_id=run_id,
            session_id=thread_id,
            agent_name=agent_name,
            delegated_task=task,
            checkpoint_namespace=checkpoint_namespace,
            status="running",
        )
        return run_id

    def interrupted(self, run_id: str) -> None:
        run = self._get_identity(run_id)
        self.store.upsert_subagent_run(**run, status="waiting_for_user")

    def failed(self, run_id: str, error: BaseException) -> None:
        run = self._get_identity(run_id)
        self.store.upsert_subagent_run(
            **run,
            status="failed",
            error_text=repr(error),
        )

    def completed(self, run_id: str, result: Any) -> None:
        messages = result.get("messages", []) if isinstance(result, dict) else []
        self.store.complete_subagent_run(run_id, serialize_messages(list(messages)))

    def _get_identity(self, run_id: str) -> dict[str, str]:
        with self.store._connect() as conn:
            row = conn.execute(
                "SELECT * FROM subagent_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Subagent run {run_id!r} was not registered.")
        return {
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "agent_name": row["agent_name"],
            "delegated_task": row["delegated_task"],
            "checkpoint_namespace": row["checkpoint_namespace"],
        }


__all__ = ["GraphInterrupt", "SubagentRunRecorder"]
