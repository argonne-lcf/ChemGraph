"""Usage printing uses counters only, with no additional model requests."""

import io

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


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
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


def test_one_shot_prints_usage_after_answer(monkeypatch, tmp_path, terminal):
    import importlib
    cli = importlib.import_module("chemgraph.cli.main")
    agent = _agent(monkeypatch, tmp_path, enable_memory=False,
                   workflow=graph(FakeMessagesListChatModel(responses=[answer()])))
    monkeypatch.setattr(cli, "initialize_agent", lambda *args, **kwargs: agent)
    monkeypatch.setattr(cli, "console", commands.console)
    monkeypatch.setattr(cli, "format_response", lambda *args, **kwargs: commands.console.print("ANSWER"))
    parser = cli.create_argument_parser()
    monkeypatch.setattr(cli.sys, "argv", ["chemgraph", "run", "-q", "test"])
    args = parser.parse_args(["run", "-q", "test"])
    cli._handle_run(args)
    output = terminal.getvalue()
    assert output.count("Tokens:") == 1
    assert output.index("ANSWER") < output.index("Tokens:")


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
    repl()
    assert terminal.getvalue().count("Session tokens:") == 1
    assert "Session tokens: 30 input · 6 output · 36 total" in terminal.getvalue()
