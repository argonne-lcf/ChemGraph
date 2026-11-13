"""Offline coverage of CLI display across failures, checkpoints, and resumes."""

import asyncio
import io
from types import SimpleNamespace

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt
from rich.console import Console

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.cli import commands, formatting
from tests.test_llm_agent import _prepared
from tests.test_main_agent_cli import _turn_result


@pytest.fixture
def terminal(monkeypatch):
    output = io.StringIO()
    console = Console(file=output, width=40, color_system=None)
    monkeypatch.setattr(commands, "console", console)
    monkeypatch.setattr(formatting, "console", console)
    monkeypatch.setattr(commands.time, "sleep", lambda _: None)
    return output


@pytest.mark.parametrize("error", [RuntimeError, KeyboardInterrupt, asyncio.CancelledError])
def test_streamed_answer_survives_failure_and_cancellation(terminal, error):
    agent = SimpleNamespace(message_renderer=None)

    async def run(*args, **kwargs):
        query = HumanMessage("query", id="q")
        agent.message_renderer([query])
        agent.message_renderer([query, AIMessage("Streamed answer survives.", id="a")])
        raise error("after streaming")

    agent.run = run
    if error is RuntimeError:
        assert commands.run_query(agent, "query") is None
    else:
        with pytest.raises(error):
            commands.run_query(agent, "query")
    assert terminal.getvalue().count("Streamed answer survives.") == 1
    assert agent.message_renderer is None


@pytest.mark.parametrize("verbose", [False, True])
def test_cold_renderer_ignores_history_and_retains_new_idless_messages(terminal, monkeypatch, verbose):
    monkeypatch.setattr(HumanMessage, "pretty_print", lambda m: commands.console.print(m.content))
    monkeypatch.setattr(AIMessage, "pretty_print", lambda m: commands.console.print(m.content))
    renderer = formatting.StreamMessageRenderer(commands.console, verbose=verbose)
    renderer.begin_turn()
    history = [HumanMessage("old query"), AIMessage("old answer"), HumanMessage("summary")]
    query = HumanMessage("new query")
    renderer.render([*history, query])
    answer = AIMessage("new answer")
    renderer.render([*history, query, answer])
    # Retained ID-less messages move when old history is removed.
    renderer.render([query, answer])
    renderer.render([query, answer, HumanMessage("generated JSON"), AIMessage("new answer")])
    renderer.finish()
    output = terminal.getvalue()
    assert "old query" not in output and "old answer" not in output and "summary" not in output
    assert ("new query" in output) is verbose
    assert output.count("new answer") == 2 and "generated JSON" in output


@pytest.mark.parametrize("return_option", ["state", "last_message"])
def test_real_graph_prints_full_prose_before_prompt_and_deduplicates_results(
    terminal, monkeypatch, tmp_path, return_option,
):
    words = [f"token{i:03}" for i in range(100)]
    prose = " ".join(words) + "\n\nMore context.\n\nLast paragraph.\x1b[2J"
    graph = StateGraph(MessagesState)
    graph.add_node("plan", lambda _: {"messages": [AIMessage(prose)]})

    def ask(_state):
        interrupt("Which method?")
        return {"messages": [AIMessage("Final response."), HumanMessage('{"positions": [[0, 0, 0]]}')]}

    graph.add_node("ask", ask)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "ask")
    graph.add_edge("ask", END)
    workflow = graph.compile(checkpointer=InMemorySaver())
    monkeypatch.setattr("chemgraph.agent.llm_agent.construct_single_agent_graph", lambda *a, **k: workflow)
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", lambda **k: _prepared())

    def prompt(_payload):
        output = terminal.getvalue()
        assert all(word in output for word in words) and "Last paragraph." in output
        assert "\x1b" not in output
        commands.console.print("CLARIFICATION PROMPT")
        return "yes"

    monkeypatch.setattr(commands, "_prompt_for_interrupt", prompt)
    agent = ChemGraph(return_option=return_option, enable_memory=False, log_dir=str(tmp_path))
    result = commands.run_query(agent, "query", thread_id=7)
    assert result is not None
    output = terminal.getvalue()
    assert output.count("token000") == 1 and output.count("Final response.") == 1
    assert output.count('"positions"') == 1
    assert output.index("Last paragraph.") < output.index("CLARIFICATION PROMPT")
    # The answer reaches format_response when the result contains it.
    assert ("ChemGraph Response" in output) is (return_option == "state")
    assert "Molecular Structure Data" in output


@pytest.mark.parametrize("session", [False, True], ids=["deep-agent", "main-agent"])
def test_cancelled_operation_closes_callbacks_before_the_next_operation(terminal, session):
    callbacks, traced = [], []
    existing = BaseCallbackHandler()
    original_config = {"configurable": {"thread_id": "thread"}, "callbacks": [existing]}

    def trace(event, payload):
        traced.append(event)
    owner = SimpleNamespace(
        thread_id="thread", config=original_config, workflow_type="deep_agent", on_event=trace,
    )

    async def operation(*args, **kwargs):
        assert owner.config["callbacks"][0] is existing
        callback = owner.config["callbacks"][-1]._on_event if session else owner.on_event
        callbacks.append(callback)
        if len(callbacks) == 2:
            # A worker from the cancelled run reports after the next run began.
            callbacks[0]("tool_call_started", {"call_id": "late", "tool_name": "late", "subagent_name": "worker"})
            callbacks[0]("tool_call_finished", {"call_id": "old", "result": "stale result"})
        callback("tool_call_started", {
            "call_id": "old" if len(callbacks) == 1 else "new",
            "tool_name": "read_file", "arguments": {"file_path": "/file"}, "subagent_name": "worker",
        })
        if len(callbacks) == 1:
            raise KeyboardInterrupt()
        callback("tool_call_finished", {"call_id": "new", "result": "fresh result"})
        return _turn_result() if session else None

    owner.run = owner.retry = owner.restore = operation
    with pytest.raises(KeyboardInterrupt):
        if session:
            commands.run_main_agent_query(owner, "query")
        else:
            commands.run_query(owner, "query")
    assert owner.config is original_config and owner.on_event is trace
    terminal.truncate(0)
    terminal.seek(0)
    if session:
        commands.retry_main_agent_session(owner)
        commands.restore_main_agent_session(owner)
    else:
        commands.run_query(owner, "next query")
    output = terminal.getvalue()
    assert "#" not in output and "late" not in output and "stale" not in output
    assert "fresh result" in output
    assert [e[2] for e in formatting.tool_output_log.entries()] == ["fresh result"]
    assert owner.config is original_config and original_config["callbacks"] == [existing]
    assert owner.on_event is trace
    if not session:
        assert "tool_call_started" in traced and "tool_call_finished" in traced

