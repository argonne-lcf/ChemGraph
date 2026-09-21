"""Hermetic usage accounting across providers, storage, and graph execution."""

from concurrent.futures import ThreadPoolExecutor
import uuid

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, LLMResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import interrupt

from chemgraph.agent.main_session import MainAgentSession
from chemgraph.agent.usage import (
    UsageCollector, add_callbacks, combine_usage, normalize_usage, response_usage, session_usage,
)
from chemgraph.memory.schemas import SessionMessage
from chemgraph.memory.store import SessionStore


def answer(text="done", inputs=10, outputs=2):
    return AIMessage(content=text, usage_metadata={
        "input_tokens": inputs, "output_tokens": outputs, "total_tokens": inputs + outputs,
    })


def response(message=None):
    return LLMResult(generations=[[ChatGeneration(message=message or answer())]])


@pytest.mark.parametrize("raw,canonical,expected", [
    ({"prompt_tokens": 10, "completion_tokens": 2}, False, (10, 2, 12)),
    ({"inputTokens": 10, "outputTokens": 2}, False, (10, 2, 12)),
    ({"input_tokens": 10}, False, (10, None, None)),
    ({"output_tokens": 2}, False, (None, 2, None)),
    ({"total_tokens": 12}, False, (None, None, 12)),
    ({"input_tokens": True, "output_tokens": -1}, False, (None, None, None)),
    ({"input_tokens": 0, "output_tokens": 0}, False, (0, 0, 0)),
    ({"input_tokens": 10, "cache_read_input_tokens": 20, "cache_creation_input_tokens": 5,
      "output_tokens": 2}, False, (35, 2, 37)),
    ({"input_tokens": 35, "input_token_details": {"cache_read": 20},
      "output_tokens": 2}, True, (35, 2, 37)),
])
def test_usage_normalization_preserves_unknown(raw, canonical, expected):
    counts = normalize_usage(raw, canonical=canonical)
    assert tuple(counts[key] for key in ("input_tokens", "output_tokens", "total_tokens")) == expected


def test_details_are_subsets_not_extra_tokens():
    usage = normalize_usage({
        "prompt_tokens": 100, "completion_tokens": 20,
        "prompt_tokens_details": {"cached_tokens": 80},
        "completion_tokens_details": {"reasoning_tokens": 15},
    })
    assert usage == {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
                     "cached_input_tokens": 80, "reasoning_output_tokens": 15}


def test_message_usage_precedes_provider_fallback():
    result = response()
    result.llm_output = {"token_usage": {"total_tokens": 999}}
    assert response_usage(result)["total_tokens"] == 12


def test_multiple_prompts_not_alternative_generations_are_added():
    result = LLMResult(generations=[
        [ChatGeneration(message=answer()), ChatGeneration(message=answer())],
        [ChatGeneration(message=answer(inputs=20))],
    ])
    assert response_usage(result)["total_tokens"] == 34
    result.generations[1] = [ChatGeneration(message=AIMessage(content="unknown"))]
    assert response_usage(result)["total_tokens"] == 12
    assert response_usage(result)["partial"] is True


@pytest.fixture
def store(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    store.create_session("session", "fake", "single_agent")
    return store


def complete(collector, key=None, message=None, **metadata):
    key = key or uuid.uuid4()
    collector.on_chat_model_start({}, [], run_id=key, metadata=metadata)
    collector.on_llm_end(response(message), run_id=key)
    return key


def test_calls_are_durable_before_finalization_and_deduplicated(store):
    collector = UsageCollector("session", "thread", store=store)
    key = complete(collector)
    collector.on_llm_end(response(), run_id=key)
    reopened = SessionStore(store.db_path)
    assert reopened.get_usage("session")["total_tokens"] == 12
    assert reopened.get_usage("session")["call_count"] == 1
    complete(collector, message=AIMessage(content="no usage"))
    totals = reopened.get_usage("session", collector.turn_id)
    assert totals["total_tokens"] == 12
    assert totals["incomplete_calls"] == 1
    assert totals["partial"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reported", [False, True])
async def test_failed_stream_preserves_usage(store, monkeypatch, asynchronous, reported):
    def stream(*args, **kwargs):
        yield ChatGenerationChunk(message=AIMessageChunk(
            content="partial", usage_metadata=answer().usage_metadata if reported else None,
        ))
        raise RuntimeError("stream disconnected")

    monkeypatch.setattr(FakeMessagesListChatModel, "_stream", stream)
    collector = UsageCollector("session", "thread", store=store)
    model = FakeMessagesListChatModel(responses=[answer()])
    with pytest.raises(RuntimeError, match="stream disconnected"):
        if asynchronous:
            async for _ in model.astream("test", config={"callbacks": [collector]}):
                pass
        else:
            list(model.stream("test", config={"callbacks": [collector]}))
    expected = 12 if reported else None
    assert collector.summary["total_tokens"] == expected
    reopened = SessionStore(store.db_path)
    totals = reopened.get_usage("session")
    assert totals["total_tokens"] == expected
    assert totals["call_count"] == 1 and totals["partial"] is True
    record, = reopened.usage_records("session")
    assert record["status"] == "failed" and record["complete"] is False
    if reported:
        assert record["raw_usage"] == [answer().usage_metadata]


@pytest.mark.parametrize("adapter_complete", [None, False, True])
def test_repeated_errors_preserve_usage_and_adapter_snapshots(store, adapter_complete):
    collector = UsageCollector("session", "thread", store=store)
    key = uuid.uuid4()
    collector.on_chat_model_start({}, [], run_id=key)
    raw = {"total": {"inputTokens": 20, "outputTokens": 4, "totalTokens": 24}}
    if adapter_complete is not None:
        collector.on_custom_event("chemgraph_usage", {
            "call_id": str(key), "counts": normalize_usage(raw["total"]),
            "raw_usage": raw, "complete": adapter_complete,
        })
    for result in (response(), response(), None):
        collector.on_llm_error(RuntimeError("failed"), run_id=key, response=result)
    totals = store.get_usage("session")
    assert totals["total_tokens"] == (12 if adapter_complete is None else 24)
    assert totals["call_count"] == 1
    assert totals["partial"] is (adapter_complete is not True)
    record, = store.usage_records("session")
    assert record["raw_usage"] == ([answer().usage_metadata] if adapter_complete is None else raw)


def test_concurrent_workers_share_one_turn(store):
    collector = UsageCollector("session", "thread", store=store)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: complete(collector, chemgraph_subagent=f"worker-{i}"), range(12)))
    assert collector.summary["total_tokens"] == 144
    assert store.get_usage("session")["call_count"] == 12
    records = store.latest_usage_turn("session", "thread")["records"]
    assert {r["worker"] for r in records} == {f"worker-{i}" for i in range(12)}


def test_restore_retry_and_new_query_totals(store):
    first = UsageCollector("session", "thread", store=store)
    complete(first)
    first.finish("failed")
    restored = store.latest_usage_turn("session", "thread")
    retry = UsageCollector("session", "thread", store=store,
                           turn_id=restored["turn_id"], records=restored["records"])
    complete(retry)
    assert retry.summary["total_tokens"] == 24
    second = UsageCollector("session", "thread", store=store)
    complete(second)
    assert second.summary["total_tokens"] == 12
    assert second.turn_id != first.turn_id
    assert store.get_usage("session")["total_tokens"] == 36
    assert store.get_usage("session", first.turn_id)["total_tokens"] == 24
    store.delete_session("session")
    with store._connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM model_usage").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM usage_turns").fetchone()[0] == 0


def test_old_sessions_are_unknown_but_recorded_no_call_turn_is_zero(store):
    assert store.get_usage("session")["total_tokens"] is None
    UsageCollector("session", "thread", store=store)
    assert store.get_usage("session")["total_tokens"] == 0


def test_storage_failure_does_not_stop_accounting(store, monkeypatch, caplog):
    def fail(*args):
        raise OSError("disk full")
    monkeypatch.setattr(store, "save_usage_call", fail)
    collector = UsageCollector("session", "thread", store=store)
    complete(collector)
    complete(collector)
    collector.finish("failed")
    assert collector.summary["total_tokens"] == 24
    assert caplog.text.count("Could not persist token usage") == 1


def graph(model, *, pause=False, fail=False):
    builder = StateGraph(MessagesState)
    builder.add_node("model", lambda state: {"messages": [model.invoke(state["messages"])]})
    def finish(state):
        if pause:
            interrupt("continue?")
        if fail:
            raise RuntimeError("forced graph failure")
        return {}
    builder.add_node("finish", finish)
    builder.add_edge(START, "model")
    builder.add_edge("model", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=InMemorySaver())


@pytest.mark.asyncio
async def test_main_session_usage_survives_approval_and_restoration(store):
    model = FakeMessagesListChatModel(responses=[answer()])
    workflow = graph(model, pause=True)
    session = MainAgentSession(workflow, thread_id="session", session_store=store)
    paused = await session.run("test")
    assert paused.usage["total_tokens"] == 12
    assert paused.status == "waiting_for_user"
    restored = MainAgentSession(workflow, thread_id="session", session_store=store)
    await restored.restore()
    result = await restored.resume("yes")
    assert result.usage["total_tokens"] == 12
    assert result.usage["turn_id"] == paused.usage["turn_id"]
    assert result.status == "completed"
    next_turn = await restored.run("again")
    assert next_turn.usage["total_tokens"] == 12
    assert next_turn.usage["turn_id"] != paused.usage["turn_id"]
    assert store.get_usage("session")["total_tokens"] == 24


@pytest.mark.asyncio
async def test_failure_persists_usage_without_dashboard(store):
    model = FakeMessagesListChatModel(responses=[answer()])
    session = MainAgentSession(graph(model, fail=True), thread_id="session", session_store=store)
    with pytest.raises(RuntimeError, match="forced graph failure"):
        await session.run("test")
    assert session.last_usage["total_tokens"] == 12
    assert store.get_usage("session")["total_tokens"] == 12


@pytest.mark.asyncio
async def test_memory_disabled_and_standalone_turn(monkeypatch, store):
    from chemgraph.agent import turn
    model = FakeMessagesListChatModel(responses=[answer()])
    monkeypatch.setattr(turn, "_load_turn_llm", lambda **kwargs: model)
    monkeypatch.setattr(turn, "construct_single_agent_graph", lambda *args, **kwargs: graph(model))
    result = await turn.run_turn(query="test", thread_id="stateless")
    assert result.usage["total_tokens"] == 12
    assert store.get_session("stateless") is None
    result = await turn.run_turn(query="test", thread_id="durable", session_store=store)
    assert result.usage["total_tokens"] == 12
    assert store.get_usage("durable")["total_tokens"] == 12


def test_existing_callback_manager_is_not_mutated():
    from langchain_core.callbacks.manager import CallbackManager
    manager = CallbackManager([])
    collector = UsageCollector("s", "t")
    new = add_callbacks({"callbacks": manager}, [collector])
    assert manager.handlers == []
    assert new["callbacks"].handlers == [collector]


@pytest.mark.asyncio
async def test_standalone_setup_failure_finishes_usage_turn(monkeypatch, store):
    from chemgraph.agent import turn
    def fail(**kwargs):
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(turn, "_load_turn_llm", fail)
    with pytest.raises(RuntimeError, match="model unavailable"):
        await turn.run_turn(query="test", thread_id="session", session_store=store)
    assert store.latest_usage_turn("session", "session")["status"] == "failed"
    assert store.get_usage("session")["total_tokens"] == 0


def test_cancellation_retains_partial_adapter_counts():
    collector = UsageCollector("s", "t")
    collector.on_custom_event("chemgraph_usage", {
        "call_id": "call", "counts": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
    })
    collector.finish("cancelled")
    assert collector.summary["total_tokens"] == 12
    assert collector.summary["partial"] is True


@pytest.mark.asyncio
async def test_nested_worker_usage_is_attributed_once(store):
    from chemgraph.graphs.main_agent import construct_main_agent_graph
    from tests.test_main_agent import _ScriptedChatModel
    delegate = answer(inputs=20)
    delegate.tool_calls = [{"name": "task", "type": "tool_call", "id": "delegate", "args": {
        "subagent_type": "worker", "description": "answer the question",
    }}]
    supervisor = _ScriptedChatModel(responses=[delegate, answer(inputs=30)])
    worker = graph(FakeMessagesListChatModel(responses=[answer()]))
    workflow = construct_main_agent_graph(
        supervisor, subagents=[{"name": "worker", "description": "test", "runnable": worker}],
        checkpointer=InMemorySaver(),
    )
    session = MainAgentSession(workflow, thread_id="session", session_store=store)
    result = await session.run("delegate this")
    assert result.usage["total_tokens"] == 66
    assert result.usage["call_count"] == 3
    assert sum(r["worker"] == "worker" for r in store.usage_records("session")) == 1


@pytest.mark.asyncio
async def test_agent_usage_and_limit_survive_manual_approval(monkeypatch, tmp_path):
    from chemgraph.agent.llm_agent import HumanInputRequired
    from langgraph.types import Command
    from tests.test_deep_agent_review import _agent
    model = FakeMessagesListChatModel(responses=[answer()])
    agent = _agent(monkeypatch, tmp_path, workflow=graph(model, pause=True))
    config = {"configurable": {"thread_id": "usage"}, "recursion_limit": 17}
    with pytest.raises(HumanInputRequired) as caught:
        await agent.run("test", config=config)
    assert "callbacks" not in config
    resume_config = caught.value.resume_config
    assert resume_config["recursion_limit"] == 17
    await agent.workflow.ainvoke(Command(resume="yes"), config=resume_config)
    assert agent.last_usage["total_tokens"] == 12
    assert agent.session_store.get_usage(agent.session_id)["total_tokens"] == 12


@pytest.mark.asyncio
async def test_agent_failure_keeps_original_exception_and_usage(monkeypatch, tmp_path):
    from tests.test_deep_agent_review import _agent
    agent = _agent(monkeypatch, tmp_path, workflow=graph(
        FakeMessagesListChatModel(responses=[answer()]), fail=True,
    ))
    with pytest.raises(RuntimeError, match="forced graph failure"):
        await agent.run("test")
    assert agent.last_usage["total_tokens"] == 12
    assert agent.session_store.get_usage(agent.session_id)["total_tokens"] == 12
    assert list((tmp_path / "logs").glob("state_*.json"))


@pytest.mark.asyncio
async def test_session_totals_include_prior_turns_with_memory_disabled(monkeypatch, tmp_path):
    from tests.test_deep_agent_review import _agent
    agent = _agent(monkeypatch, tmp_path, enable_memory=False,
                   workflow=graph(FakeMessagesListChatModel(responses=[answer()])))
    await agent.run("first")
    await agent.run("second")
    assert agent.last_usage["total_tokens"] == 12
    assert agent.session_usage["total_tokens"] == 24
    assert not (tmp_path / "memory.db").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("durable", [False, True])
async def test_legacy_checkpoint_history_stays_partial_after_new_turn(store, durable):
    workflow = graph(FakeMessagesListChatModel(responses=[answer()]))
    await workflow.ainvoke({"messages": [("human", "legacy")]},
                          config={"configurable": {"thread_id": "session"}})
    session = MainAgentSession(workflow, thread_id="session", session_store=store if durable else None)
    await session.restore()
    assert session.session_usage["history_unaccounted"] is True
    assert session.session_usage["total_tokens"] is None
    assert session.session_usage["call_count"] == 0
    result = await session.run("new query")
    assert result.usage["partial"] is False
    assert session.session_usage["total_tokens"] == 12
    assert session.session_usage["partial"] is True
    assert session.session_usage["incomplete_calls"] == 0
    if durable:
        assert store.get_usage("session")["history_unaccounted"] is True
        assert store.get_usage("session", result.usage["turn_id"])["partial"] is False
        session = MainAgentSession(workflow, thread_id="session", session_store=SessionStore(store.db_path))
    await session.restore()
    totals = combine_usage([session.session_usage])
    assert totals["total_tokens"] == 12
    assert totals["history_unaccounted"] is True
    assert totals["partial"] is True


@pytest.mark.parametrize("recorded", [False, True])
def test_usage_history_migration_is_conservative_and_sticky(store, recorded):
    if recorded:
        complete(UsageCollector("session", "thread", store=store))
    store.save_messages("session", [SessionMessage(role="human", content="historical query")])
    store.create_session("empty", "fake", "main_agent")
    with store._connect() as conn:
        conn.execute("ALTER TABLE sessions DROP COLUMN history_unaccounted")
    migrated = SessionStore(store.db_path)
    assert migrated.get_usage("session")["history_unaccounted"] is True
    assert migrated.get_usage("session")["partial"] is True
    assert migrated.get_usage("session")["total_tokens"] == (12 if recorded else None)
    assert migrated.get_usage("empty")["history_unaccounted"] is False
    complete(UsageCollector("session", "thread", store=migrated))
    reopened = SessionStore(store.db_path)
    assert reopened.get_usage("session")["history_unaccounted"] is True
    assert reopened.get_usage("session")["total_tokens"] == (24 if recorded else 12)
    empty = UsageCollector("empty", "thread", store=reopened)
    assert reopened.get_usage("empty", empty.turn_id)["total_tokens"] == 0
    assert reopened.get_usage("empty")["partial"] is False


def test_first_usage_turn_preserves_existing_transcript_gap(store):
    store.save_messages("session", [SessionMessage(role="human", content="legacy")])
    assert store.get_usage("session")["history_unaccounted"] is True
    collector = UsageCollector("session", "thread", store=store)
    complete(collector)
    assert store.get_usage("session")["partial"] is True
    assert store.get_usage("session", collector.turn_id)["partial"] is False


@pytest.mark.asyncio
async def test_legacy_coverage_survives_storage_failure(store, monkeypatch):
    workflow = graph(FakeMessagesListChatModel(responses=[answer()]))
    await workflow.ainvoke({"messages": [("human", "legacy")]},
                          config={"configurable": {"thread_id": "session"}})
    def fail(*_):
        raise OSError("storage unavailable")
    for method in ("mark_usage_history_unaccounted", "create_usage_turn", "save_usage_call",
                   "update_usage_turn", "usage_history_unaccounted", "usage_records"):
        monkeypatch.setattr(store, method, fail)
    session = MainAgentSession(workflow, thread_id="session", session_store=store)
    await session.restore()
    assert session.session_usage["total_tokens"] is None
    await session.run("new query")
    await session.restore()
    assert session.session_usage["total_tokens"] == 12
    assert session.session_usage["history_unaccounted"] is True
    assert session.session_usage["partial"] is True


def test_coverage_read_failure_does_not_hide_durable_counts(store, monkeypatch):
    complete(UsageCollector("session", "thread", store=store))
    def fail(*_):
        raise OSError("coverage unavailable")
    monkeypatch.setattr(store, "usage_history_unaccounted", fail)
    summary = session_usage([], store, "session")
    assert summary["total_tokens"] == 12
    assert summary["partial"] is True
