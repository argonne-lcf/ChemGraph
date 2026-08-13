"""Coordinated deletion across readable sessions and graph checkpoints."""

from __future__ import annotations

from typing import Any

from chemgraph.memory.store import SessionStore


def delete_durable_session(
    store: SessionStore,
    session_id: str,
    *,
    checkpointer: Any | None = None,
) -> bool:
    """Delete checkpoints first, then transactionally remove readable records."""
    session = store.get_session(session_id)
    if session is None:
        return False
    if session.graph_config is None:
        return store.delete_session(session.session_id)

    if checkpointer is not None:
        raise RuntimeError(
            "Caller-supplied async checkpointers must be deleted from their owner loop."
        )
    if session.checkpoint_backend not in {None, "AsyncSqliteSaver", "memory"}:
        raise RuntimeError(
            "This session uses an external checkpointer; delete it through the "
            "caller-owned saver before removing the session record."
        )
    if session.checkpoint_backend == "memory":
        raise RuntimeError(
            "The session refers to a process-local checkpointer that is no longer active."
        )

    from chemgraph.cli.checkpoint_runtime import CheckpointRuntime

    runtime = CheckpointRuntime()
    try:
        saver = runtime.open_sqlite(session.checkpoint_db)
        runtime.delete_thread(saver, session.session_id)
    finally:
        runtime.close()
    return store.delete_session(session.session_id)


__all__ = ["delete_durable_session"]
