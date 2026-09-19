"""Usage printing uses counters only, with no additional model requests."""

import io
import gc
import weakref
from asyncio import CancelledError

import pytest
from rich.console import Console
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt

from chemgraph.agent.main_session import MainAgentSession
from chemgraph.cli import commands
from chemgraph.cli.formatting import format_token_usage
from chemgraph.memory.store import SessionStore
from tests.test_usage import answer, graph
from tests.test_deep_agent_review import _agent


@pytest.fixture
def terminal(monkeypatch):
    output = io.StringIO()
    console = Console(file=output, width=180, color_system=None)
    monkeypatch.setattr(commands, "console", console)
    monkeypatch.setattr(commands.time, "sleep", lambda *_: None)
    return output


def test_format_usage_unknown_partial_and_subsets():
    counts = {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
              "cached_input_tokens": 80, "reasoning_output_tokens": 5, "call_count": 1}
    text = format_token_usage(counts).plain
    assert text.startswith("Tokens: 100 input · 20 output · 120 total")
    assert "included: 80 cached input; 5 reasoning output" in text
    counts.update(input_tokens=None, output_tokens=None, total_tokens=None)
    assert format_token_usage(counts).plain.startswith("Tokens: unavailable")
    counts.update(input_tokens=100, partial=True, incomplete_calls=1)
    assert "Tokens (partial): 100 input · unknown output · unknown total" in format_token_usage(counts).plain


def test_standalone_approval_counts_calls_on_both_sides(monkeypatch, tmp_path, terminal):
    model = FakeMessagesListChatModel(responses=[answer(), answer(), answer()])
    builder = StateGraph(MessagesState)
    builder.add_node("first", lambda state: {"messages": [model.invoke(state["messages"])]})
    def second(state):
        interrupt("continue?")
        return {"messages": [model.invoke(state["messages"])]}
    builder.add_node("second", second)
    builder.add_edge(START, "first")
    builder.add_edge("first", "second")
    builder.add_edge("second", END)
    agent = _agent(monkeypatch, tmp_path, workflow=builder.compile(checkpointer=InMemorySaver()))
    monkeypatch.setattr(commands, "_prompt_for_interrupt", lambda *_: "yes")
    result = commands.run_query(agent, "test")
    assert result.content == "done"
    assert "Tokens:" not in terminal.getvalue()  # caller renders after the answer
    commands.print_token_usage(agent)
    assert terminal.getvalue().count("Tokens:") == 1
    assert "24 total" in terminal.getvalue()
    assert model.i == 2  # no extra invocation to produce the usage line
    assert agent.last_usage["call_count"] == 2
    state = agent.get_state({"configurable": {"thread_id": agent.last_usage["thread_id"]}})
    assert all("Tokens:" not in m.content for m in state.get("messages", []))


@pytest.mark.parametrize("main", [False, True])
def test_cli_failure_prints_known_usage_once(monkeypatch, tmp_path, terminal, main):
    workflow = graph(FakeMessagesListChatModel(responses=[answer()]), fail=True)
    if main:
        owner = MainAgentSession(workflow)
        result = commands.run_main_agent_query(owner, "test")
    else:
        owner = _agent(monkeypatch, tmp_path, workflow=workflow)
        result = commands.run_query(owner, "test")
    assert result is None
    assert terminal.getvalue().count("Tokens:") == 1
    assert "12 total" in terminal.getvalue()


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt, CancelledError])
def test_cli_cancelled_review_prints_usage_without_resuming(monkeypatch, tmp_path, terminal, error):
    model = FakeMessagesListChatModel(responses=[answer(), answer()])
    agent = _agent(monkeypatch, tmp_path, workflow=graph(model, pause=True))
    def cancel(*args):
        raise error()
    monkeypatch.setattr(commands, "_prompt_for_interrupt", cancel)
    with pytest.raises(error):
        commands.run_query(agent, "test")
    assert terminal.getvalue().count("Tokens:") == 1
    assert "12 total" in terminal.getvalue()
    assert model.i == 1
    assert agent.session_store.latest_usage_turn(
        agent.session_id, agent.last_usage["thread_id"],
    )["status"] == "cancelled"


def test_exit_usage_counts_turns_once_even_after_retry(monkeypatch, tmp_path, terminal):
    agent = _agent(monkeypatch, tmp_path, enable_memory=False,
                   workflow=graph(FakeMessagesListChatModel(responses=[answer()])))
    @commands._report_session_usage
    def repl():
        commands.run_query(agent, "one")
        commands.print_token_usage(agent)
        commands.run_query(agent, "two")
        commands.print_token_usage(agent)
    repl()
    assert terminal.getvalue().count("Tokens:") == 2
    assert terminal.getvalue().count("Session tokens:") == 1
    assert "Session tokens: 20 input · 4 output · 24 total" in terminal.getvalue()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_one_shot_prints_usage_only_to_stderr(monkeypatch, tmp_path, terminal, capsys, outcome):
    import importlib
    cli = importlib.import_module("chemgraph.cli.main")
    agent = _agent(monkeypatch, tmp_path, enable_memory=False,
                   workflow=graph(FakeMessagesListChatModel(responses=[answer()]),
                                  fail=outcome == "failure", pause=outcome == "cancelled"))
    monkeypatch.setattr(cli, "initialize_agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr(cli, "console", commands.console)
    monkeypatch.setattr(cli, "format_response", lambda *args, **kwargs: commands.console.print("ANSWER"))
    parser = cli.create_argument_parser()
    monkeypatch.setattr(cli.sys, "argv", ["chemgraph", "run", "-q", "test"])
    args = parser.parse_args(["run", "-q", "test"])
    if outcome == "cancelled":
        def cancel(*_):
            raise KeyboardInterrupt()
        monkeypatch.setattr(commands, "_prompt_for_interrupt", cancel)
        with pytest.raises(KeyboardInterrupt):
            cli._handle_run(args)
    else:
        cli._handle_run(args)
    output = terminal.getvalue()
    captured = capsys.readouterr()
    assert "Tokens:" not in output + captured.out
    assert captured.err.count("Tokens:") == 1
    assert "12 total" in captured.err
    if outcome == "success":
        assert "ANSWER" in output


@pytest.mark.parametrize("exit_command", ["/quit", None])
def test_interactive_exit_prints_cumulative_usage(monkeypatch, tmp_path, terminal, exit_command):
    agent = _agent(monkeypatch, tmp_path, enable_memory=False,
                   workflow=graph(FakeMessagesListChatModel(responses=[answer()])))
    monkeypatch.setattr(commands, "initialize_agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr(commands, "format_response", lambda *args, **kwargs: commands.console.print("ANSWER"))
    inputs = iter(["test", "single_agent", "one", "two", exit_command])
    def ask(*args, **kwargs):
        item = next(inputs)
        if item is None:
            raise EOFError()
        return item
    monkeypatch.setattr(commands.Prompt, "ask", ask)
    commands.interactive_mode(model="test", workflow="single_agent")
    output = terminal.getvalue()
    assert output.count("Tokens:") == 2
    assert output.count("Session tokens:") == 1
    assert "Session tokens: 20 input · 4 output · 24 total" in output
    assert output.index("ANSWER") < output.index("Tokens:")


def test_resume_failure_preserves_checkpoint_and_usage(monkeypatch, tmp_path, terminal):
    from unittest.mock import AsyncMock
    agent = _agent(monkeypatch, tmp_path, workflow=graph(
        FakeMessagesListChatModel(responses=[answer()]), pause=True, fail=True,
    ))
    persist = AsyncMock(wraps=agent.apersist_run_state)
    monkeypatch.setattr(agent, "apersist_run_state", persist)
    monkeypatch.setattr(commands, "_prompt_for_interrupt", lambda *_: "yes")
    assert commands.run_query(agent, "test") is None
    assert persist.await_count == 2  # approval checkpoint and failed continuation
    assert "forced graph failure" in terminal.getvalue()
    assert terminal.getvalue().count("Tokens:") == 1
    assert agent.session_store.get_usage(agent.session_id)["total_tokens"] == 12


def test_exit_totals_include_restored_history_and_session_switch(monkeypatch, tmp_path, terminal):
    workflow = graph(FakeMessagesListChatModel(responses=[answer()]))
    store = SessionStore(str(tmp_path / "history.db"))
    first = MainAgentSession(workflow, thread_id="history", session_store=store)
    commands.run_async_callable(lambda: first.run("previous process"))
    restored = MainAgentSession(workflow, thread_id="history", session_store=store)
    other = _agent(monkeypatch, tmp_path, enable_memory=False,
                   workflow=graph(FakeMessagesListChatModel(responses=[answer()])))
    @commands._report_session_usage
    def repl():
        commands.restore_main_agent_session(restored)
        commands.run_main_agent_query(restored, "new query")
        commands.run_query(other, "different model")
        # Revisit the same durable session: replace its cumulative summary.
        commands.restore_main_agent_session(restored)
    repl()
    assert terminal.getvalue().count("Session tokens:") == 1
    assert "Session tokens: 30 input · 6 output · 36 total" in terminal.getvalue()


@pytest.mark.parametrize("prior_status", ["completed", "failed", "waiting_for_user"])
def test_rejected_main_operation_does_not_change_prior_usage(tmp_path, terminal, prior_status):
    store = SessionStore(str(tmp_path / "history.db"))
    workflow = graph(FakeMessagesListChatModel(responses=[answer()]),
                     fail=prior_status == "failed", pause=prior_status == "waiting_for_user")
    session = MainAgentSession(workflow, thread_id="history", session_store=store)
    try:
        commands.run_async_callable(lambda: session.run("first"))
    except RuntimeError:
        assert prior_status == "failed"
    before = store.latest_usage_turn("history", "history")
    assert before["status"] == prior_status
    terminal.seek(0)
    terminal.truncate()
    assert commands.run_main_agent_query(session, " ") is None
    assert "Tokens:" not in terminal.getvalue()
    assert store.latest_usage_turn("history", "history") == before
    assert session.last_usage["total_tokens"] == 12


def test_rejected_retry_after_idle_restore_preserves_completed_turn(tmp_path, terminal):
    store = SessionStore(str(tmp_path / "history.db"))
    workflow = graph(FakeMessagesListChatModel(responses=[answer()]))
    session = MainAgentSession(workflow, thread_id="history", session_store=store)
    commands.run_main_agent_query(session, "first")
    restored = MainAgentSession(workflow, thread_id="history", session_store=store)
    before = store.latest_usage_turn("history", "history")
    commands.restore_main_agent_session(restored)
    assert commands.retry_main_agent_session(restored) is None
    assert "Tokens:" not in terminal.getvalue()
    assert store.latest_usage_turn("history", "history") == before


def test_rejected_resume_keeps_pending_turn(tmp_path, terminal):
    store = SessionStore(str(tmp_path / "pending.db"))
    session = MainAgentSession(graph(FakeMessagesListChatModel(responses=[answer()]), pause=True),
                               thread_id="pending", session_store=store)
    commands.run_async_callable(lambda: session.run("first"))
    before = store.latest_usage_turn("pending", "pending")
    assert commands._run_main_agent_operation(
        session, lambda: session.resume(""), progress_description="Resuming",
    ) is None
    assert "Tokens:" not in terminal.getvalue()
    assert store.latest_usage_turn("pending", "pending") == before


def test_rejected_standalone_query_keeps_prior_turn(monkeypatch, tmp_path, terminal):
    agent = _agent(monkeypatch, tmp_path, workflow=graph(FakeMessagesListChatModel(responses=[answer()])))
    commands.run_query(agent, "first")
    before = agent.session_store.latest_usage_turn(agent.session_id, agent.last_usage["thread_id"])
    agent.recursion_limit = 0
    assert commands.run_query(agent, "second") is None
    assert "Tokens:" not in terminal.getvalue()
    assert agent.session_store.latest_usage_turn(agent.session_id, agent.last_usage["thread_id"]) == before


def test_interactive_registry_does_not_retain_replaced_owner(monkeypatch, tmp_path, terminal):
    @commands._report_session_usage
    def repl():
        agent = _agent(monkeypatch, tmp_path, enable_memory=False,
                       workflow=graph(FakeMessagesListChatModel(responses=[answer()])))
        commands.run_query(agent, "first")
        reference = weakref.ref(agent)
        del agent
        gc.collect()
        assert reference() is None
    repl()
    assert "Session tokens: 10 input · 2 output · 12 total" in terminal.getvalue()


def test_cli_retry_counts_reexecuted_calls_in_same_turn(tmp_path, terminal):
    model = FakeMessagesListChatModel(responses=[answer()])
    attempts = 0
    def node(state):
        nonlocal attempts
        attempts += 1
        result = model.invoke(state["messages"])
        if attempts == 1:
            raise RuntimeError("retry this node")
        return {"messages": [result]}
    builder = StateGraph(MessagesState)
    builder.add_node("model", node)
    builder.add_edge(START, "model")
    builder.add_edge("model", END)
    store = SessionStore(str(tmp_path / "retry.db"))
    session = MainAgentSession(builder.compile(checkpointer=InMemorySaver()), session_store=store)
    assert commands.run_main_agent_query(session, "first") is None
    turn_id = session.last_usage["turn_id"]
    result = commands.retry_main_agent_session(session)
    assert result.status == "completed"
    assert result.usage["turn_id"] == turn_id
    assert result.usage["total_tokens"] == 24
    assert store.latest_usage_turn(session.thread_id, session.thread_id)["status"] == "completed"


@pytest.mark.parametrize("new_query", [False, True])
def test_legacy_restore_is_included_in_interactive_exit_usage(tmp_path, terminal, new_query):
    store = SessionStore(str(tmp_path / "legacy.db"))
    store.create_session("legacy", "fake", "main_agent")
    workflow = graph(FakeMessagesListChatModel(responses=[answer()]))
    commands.run_async_callable(lambda: workflow.ainvoke(
        {"messages": [("human", "legacy query")]}, config={"configurable": {"thread_id": "legacy"}},
    ))
    session = MainAgentSession(workflow, thread_id="legacy", session_store=store)
    @commands._report_session_usage
    def repl():
        commands.restore_main_agent_session(session)
        if new_query:
            commands.run_main_agent_query(session, "new")
    repl()
    output = terminal.getvalue()
    assert "historical usage was not recorded" in output
    assert "incomplete usage for 0 call(s)" not in output
    expected = "Session tokens (partial): 10 input" if new_query else "Session tokens: unavailable"
    assert expected in output
