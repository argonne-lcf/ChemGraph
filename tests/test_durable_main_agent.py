"""Durable checkpoint and transcript tests for the main-agent workflow."""

import os
import sqlite3
import stat
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from chemgraph.agent.main_session import (
    IncompatibleCheckpointError,
    MainAgentSession,
    MissingCheckpointError,
)
from chemgraph.cli import checkpoint_runtime as checkpoint_runtime_module
from chemgraph.cli.checkpoint_runtime import CheckpointRuntime
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.memory.durable import delete_durable_session
from chemgraph.memory.schemas import MainAgentGraphConfig, MainAgentSessionMetadata
from chemgraph.memory.store import SessionStore
from chemgraph.memory.subagent_recorder import SubagentRunRecorder
from tests.test_main_agent import (
    _ScriptedChatModel,
    _answering_subgraph,
    _interrupting_subgraph,
    _subagent,
    _task_call,
)


def _metadata(checkpoint_db, fingerprint="topology"):
    return MainAgentSessionMetadata(
        graph_config=MainAgentGraphConfig(
            model_name="scripted",
            topology_fingerprint=fingerprint,
        ),
        checkpoint_backend="AsyncSqliteSaver",
        checkpoint_db=str(checkpoint_db),
    )


def _graph(model, saver, *, child=None, recorder=None):
    return construct_main_agent_graph(
        model,
        subagents=[_subagent(child or _answering_subgraph("child"))],
        checkpointer=saver,
        subagent_recorder=recorder,
    )


def test_restart_restores_history_and_continues(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    metadata = _metadata(checkpoint_db)

    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[AIMessage(content="first")]), saver)
    session = MainAgentSession(
        graph,
        thread_id="durable-thread",
        session_store=store,
        session_metadata=metadata,
    )
    runtime.run(lambda: session.run("first question"))
    runtime.close()

    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[AIMessage(content="second")]), saver)
    restored = MainAgentSession(
        graph,
        thread_id="durable-thread",
        session_store=store,
        session_metadata=metadata,
    )
    assert runtime.run(restored.restore).status == "completed"
    result = runtime.run(lambda: restored.run("follow up"))
    runtime.close()

    assert result.assistant_response == "second"
    saved = store.get_session("durable-thread")
    assert [message.content for message in saved.messages if message.role == "human"] == [
        "first question",
        "follow up",
    ]
    assert saved.query_count == 2


def test_restart_restores_subagent_interrupt_and_transcript(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    metadata = _metadata(checkpoint_db)

    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(
        _ScriptedChatModel(
            responses=[AIMessage(content="", tool_calls=[_task_call("task-1")])]
        ),
        saver,
        child=_interrupting_subgraph(),
        recorder=SubagentRunRecorder(store),
    )
    session = MainAgentSession(
        graph,
        thread_id="interrupt-thread",
        session_store=store,
        session_metadata=metadata,
    )
    waiting = runtime.run(lambda: session.run("calculate"))
    assert waiting.status == "waiting_for_user"
    assert store.get_session("interrupt-thread").child_runs[0].status == (
        "waiting_for_user"
    )
    runtime.close()

    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(
        _ScriptedChatModel(responses=[AIMessage(content="used EMT")]),
        saver,
        child=_interrupting_subgraph(),
        recorder=SubagentRunRecorder(store),
    )
    restored = MainAgentSession(
        graph,
        thread_id="interrupt-thread",
        session_store=store,
        session_metadata=metadata,
    )
    assert runtime.run(restored.restore).status == "waiting_for_user"
    completed = runtime.run(lambda: restored.resume("EMT"))
    runtime.close()

    assert completed.assistant_response == "used EMT"
    saved = store.get_session("interrupt-thread")
    assert len(saved.child_runs) == 1
    assert saved.child_runs[0].status == "completed"
    assert [message.role for message in saved.child_runs[0].messages] == [
        "human",
        "ai",
    ]


def test_restart_detects_failed_task_and_retry_is_idempotent(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    metadata = _metadata(checkpoint_db)

    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[RuntimeError("temporary")]), saver)
    session = MainAgentSession(
        graph,
        thread_id="failed-thread",
        session_store=store,
        session_metadata=metadata,
    )
    with pytest.raises(RuntimeError, match="temporary"):
        runtime.run(lambda: session.run("hello"))
    runtime.close()

    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[AIMessage(content="recovered")]), saver)
    restored = MainAgentSession(
        graph,
        thread_id="failed-thread",
        session_store=store,
        session_metadata=metadata,
    )
    assert runtime.run(restored.restore).status == "failed"
    assert runtime.run(restored.retry).assistant_response == "recovered"
    snapshot = runtime.run(lambda: graph.aget_state(restored.config))
    runtime.close()

    humans = [
        message.content
        for message in snapshot.values["messages"]
        if isinstance(message, HumanMessage)
    ]
    assert humans == ["hello"]
    assert store.get_session("failed-thread").query_count == 1


def test_restore_rejects_topology_mismatch(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[AIMessage(content="done")]), saver)
    session = MainAgentSession(
        graph,
        thread_id="mismatch",
        session_store=store,
        session_metadata=_metadata(checkpoint_db, "one"),
    )
    runtime.run(lambda: session.run("hello"))
    incompatible = MainAgentSession(
        graph,
        thread_id="mismatch",
        session_store=store,
        session_metadata=_metadata(checkpoint_db, "two"),
    )
    with pytest.raises(IncompatibleCheckpointError):
        runtime.run(incompatible.restore)
    runtime.close()


def test_restore_rejects_missing_checkpoint(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    metadata = _metadata(checkpoint_db)
    store.create_session(
        "missing",
        "scripted",
        "main_agent",
        session_metadata=metadata,
    )
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[]), saver)
    session = MainAgentSession(
        graph,
        thread_id="missing",
        session_store=store,
        session_metadata=metadata,
    )

    with pytest.raises(MissingCheckpointError):
        runtime.run(session.restore)
    runtime.close()


def test_parallel_direct_subagents_get_isolated_run_records(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    first = _task_call("first", "one")
    second = _task_call("second", "two")
    graph = _graph(
        _ScriptedChatModel(
            responses=[
                AIMessage(content="", tool_calls=[first, second]),
                AIMessage(content="complete"),
            ]
        ),
        saver,
        recorder=SubagentRunRecorder(store),
    )
    session = MainAgentSession(
        graph,
        thread_id="parallel",
        session_store=store,
        session_metadata=_metadata(checkpoint_db),
    )

    runtime.run(lambda: session.run("do both"))
    runtime.close()

    runs = store.get_session("parallel").child_runs
    assert len(runs) == 2
    assert {run.delegated_task for run in runs} == {"one", "two"}
    assert len({run.run_id for run in runs}) == 2
    assert all(run.status == "completed" for run in runs)


def test_child_tool_transcript_is_lossless_and_replay_idempotent(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    store.create_session(
        "root",
        "scripted",
        "main_agent",
        session_metadata=_metadata(tmp_path / "checkpoints.db"),
    )
    recorder = SubagentRunRecorder(store)
    state = {"messages": [HumanMessage(content="calculate the energy")]}
    config = {
        "configurable": {
            "thread_id": "root",
            "checkpoint_ns": "task:opaque-namespace",
        }
    }
    result = {
        "messages": [
            *state["messages"],
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "run_ase",
                        "args": {"driver": "energy"},
                        "id": "tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content='{"energy": -1.0}', tool_call_id="tool-1"),
            AIMessage(content="The energy is -1.0 eV."),
        ]
    }

    run_id = recorder.start("chemgraph", state, config)
    recorder.completed(run_id, result)
    recorder.completed(run_id, result)

    runs = store.get_session("root").child_runs
    assert len(runs) == 1
    assert [message.role for message in runs[0].messages] == [
        "human",
        "ai",
        "tool",
        "ai",
    ]
    assert runs[0].messages[1].content == "[tool calls: run_ase]"
    assert all(message.serialized_payload for message in runs[0].messages)
    assert len({message.message_id for message in runs[0].messages}) == 4


def test_unserializable_child_transcript_does_not_fail_completed_run(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    store.create_session(
        "root",
        "scripted",
        "main_agent",
        session_metadata=_metadata(tmp_path / "checkpoints.db"),
    )
    recorder = SubagentRunRecorder(store)
    state = {"messages": [HumanMessage(content="calculate")]}
    config = {
        "configurable": {
            "thread_id": "root",
            "checkpoint_ns": "task:unserializable",
        }
    }
    run_id = recorder.start("chemgraph", state, config)

    recorder.completed(
        run_id,
        {
            "messages": [
                ToolMessage(
                    content="done",
                    tool_call_id="tool-1",
                    artifact=object(),
                )
            ]
        },
    )

    run = store.get_session("root").child_runs[0]
    assert run.status == "completed"
    assert run.messages == []
    assert run.error_text == "Readable transcript unavailable: TypeError"


def test_readable_sync_failure_does_not_break_completed_turn(monkeypatch, tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[AIMessage(content="done")]), saver)
    session = MainAgentSession(
        graph,
        thread_id="store-failure",
        session_store=store,
        session_metadata=_metadata(checkpoint_db),
    )
    monkeypatch.setattr(
        store,
        "synchronize_messages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("store broke")),
    )

    result = runtime.run(lambda: session.run("hello"))
    runtime.close()

    assert result.status == "completed"
    assert store.get_session("store-failure").status == "completed"


def test_readable_registration_failure_does_not_prevent_turn(monkeypatch, tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    monkeypatch.setattr(
        store,
        "create_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("store broke")),
    )
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[AIMessage(content="done")]), saver)
    session = MainAgentSession(
        graph,
        thread_id="registration-failure",
        session_store=store,
        session_metadata=_metadata(checkpoint_db),
    )

    result = runtime.run(lambda: session.run("hello"))
    runtime.close()

    assert result.status == "completed"
    assert store.get_session("registration-failure") is None


def test_status_failure_does_not_mask_graph_error(monkeypatch, tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[RuntimeError("graph broke")]), saver)
    session = MainAgentSession(
        graph,
        thread_id="original-error",
        session_store=store,
        session_metadata=_metadata(checkpoint_db),
    )
    session._ensure_registered("hello")
    monkeypatch.setattr(
        store,
        "update_session_status",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("store broke")),
    )

    with pytest.raises(RuntimeError, match="graph broke"):
        runtime.run(lambda: session.run("hello"))
    runtime.close()


def test_all_recorder_storage_failures_are_nonfatal():
    class BrokenStore:
        def upsert_subagent_run(self, **_kwargs):
            raise RuntimeError("store broke")

    recorder = SubagentRunRecorder(BrokenStore())
    config = {
        "configurable": {
            "thread_id": "root",
            "checkpoint_ns": "task:broken-store",
        }
    }

    assert recorder.start("chemgraph", {"messages": []}, config) is None
    recorder.interrupted("missing")
    recorder.failed("missing", RuntimeError("graph broke"))
    recorder.completed("missing", {"messages": []})


def test_delete_removes_session_children_and_checkpoints(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    metadata = _metadata(checkpoint_db)
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    graph = _graph(_ScriptedChatModel(responses=[AIMessage(content="done")]), saver)
    session = MainAgentSession(
        graph,
        thread_id="delete-me",
        session_store=store,
        session_metadata=metadata,
    )
    runtime.run(lambda: session.run("hello"))
    runtime.close()

    assert delete_durable_session(store, "delete-me") is True
    assert store.get_session("delete-me") is None
    runtime = CheckpointRuntime()
    saver = runtime.open_sqlite(str(checkpoint_db))
    assert runtime.run(
        lambda: saver.aget({"configurable": {"thread_id": "delete-me"}})
    ) is None
    runtime.close()


def test_delete_removes_process_local_session_record(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    metadata = MainAgentSessionMetadata(
        graph_config=MainAgentGraphConfig(model_name="scripted"),
        checkpoint_backend="memory",
    )
    store.create_session(
        "process-local",
        "scripted",
        "main_agent",
        session_metadata=metadata,
    )

    assert delete_durable_session(store, "process-local") is True
    assert store.get_session("process-local") is None


def test_legacy_session_database_migrates_in_place(tmp_path):
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
                model_name TEXT NOT NULL, workflow_type TEXT NOT NULL,
                log_dir TEXT, query_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                role TEXT NOT NULL, content TEXT NOT NULL, tool_name TEXT,
                timestamp TEXT NOT NULL
            );
            INSERT INTO sessions VALUES
                ('legacy', 'Old', 'gpt', 'single_agent', NULL, 0,
                 '2026-01-01T00:00:00', '2026-01-01T00:00:00');
            """
        )

    migrated = SessionStore(str(db_path)).get_session("legacy")

    assert migrated.status == "completed"
    assert migrated.graph_config is None


@pytest.mark.skipif(os.name != "posix", reason="POSIX modes are platform-specific")
def test_local_database_files_are_private(tmp_path):
    session_db = tmp_path / "sessions.db"
    checkpoint_db = tmp_path / "checkpoints.db"
    SessionStore(str(session_db))
    runtime = CheckpointRuntime()
    runtime.open_sqlite(str(checkpoint_db))
    runtime.close()

    assert stat.S_IMODE(session_db.stat().st_mode) == 0o600
    assert stat.S_IMODE(checkpoint_db.stat().st_mode) == 0o600


def test_busy_runtime_shutdown_is_bounded_and_finishes_on_owner_thread(
    monkeypatch, caplog
):
    owner_blocked = threading.Event()
    release_owner = threading.Event()

    class BlockingConnection:
        async def close(self):
            import asyncio

            def block_owner():
                owner_blocked.set()
                release_owner.wait()

            asyncio.get_running_loop().call_soon(block_owner)

    monkeypatch.setattr(checkpoint_runtime_module, "_CLOSE_TIMEOUT_SECONDS", 0.03)
    runtime = CheckpointRuntime()
    runtime._connections["blocking"] = BlockingConnection()
    close_thread = threading.Thread(target=runtime.close)

    try:
        close_thread.start()
        assert owner_blocked.wait(timeout=1.0)
        close_thread.join(timeout=1.0)

        assert not close_thread.is_alive()
        assert runtime._thread.is_alive()
        assert "still running after shutdown" in caplog.text
    finally:
        release_owner.set()
        close_thread.join(timeout=1.0)

    runtime._thread.join(timeout=1.0)
    assert not runtime._thread.is_alive()


def test_close_sqlite_releases_cached_saver(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    runtime = CheckpointRuntime()
    first = runtime.open_sqlite(str(checkpoint_db))

    runtime.close_sqlite(str(checkpoint_db))
    second = runtime.open_sqlite(str(checkpoint_db))
    runtime.close()

    assert second is not first


def test_runtime_can_be_closed_from_owner_loop():
    runtime = CheckpointRuntime()

    async def close_runtime():
        runtime.close()

    runtime.run(close_runtime)
    runtime._thread.join(timeout=0.5)

    assert not runtime._thread.is_alive()
    runtime.close()
