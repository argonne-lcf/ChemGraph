"""Exercise the execution boundary through real graphs, without live models."""

import asyncio
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from chemgraph.execution.scoped import (
    ScopedExecutionError,
    execution_context,
    result_artifacts,
)
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.registry import ToolRegistry
from chemgraph.tools.ase_tools import run_ase
from tests.test_main_agent import _ScriptedChatModel
from tests.test_scoped_execution import Controller, scope


def call(name, args, identifier):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": identifier}]
    )


def graph_for(mode, saver, *, policy=None):
    energy = call(
        "run_ase",
        {
            "params": {
                "input_structure_file": "h2.xyz",
                "output_results_file": "energy.json",
                "calculator": {"calculator_type": "emt"},
                "driver": "energy",
            }
        },
        "energy",
    )
    responses = [energy, AIMessage(content="Done")]
    options = {}
    if mode == "registry":
        responses.insert(0, call("load_tools", {"names": ["run_ase"]}, "load"))
    elif mode == "attached":
        options = {"main_tools": [run_ase], "tool_registry": ToolRegistry([])}
    else:
        responses = [
            call("load_agents", {"names": ["single_agent"]}, "load"),
            call(
                "task",
                {"subagent_type": "single_agent", "description": "EMT energy"},
                "task",
            ),
            *responses,
            AIMessage(content="Parent done"),
        ]
    return construct_main_agent_graph(
        _ScriptedChatModel(responses=responses),
        checkpointer=saver,
        discover_skills=False,
        interrupt_on=policy,
        **options,
    )


def seed(context):
    (context.workspace / "h2.xyz").write_text("2\nEMT test\nH 0 0 0\nH 0 0 0.9\n")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["registry", "attached", "delegated"])
async def test_native_paths_journal_replay_and_restore(tmp_path, monkeypatch, mode):
    from chemgraph.tools import ase_tools

    context = scope(tmp_path)
    seed(context)
    original = ase_tools.run_ase_core
    invoked = []

    def calculate(params):
        assert context.controller.calls[-1] == ("alice", "energy")
        invoked.append(params)
        return original(params)

    monkeypatch.setattr(ase_tools, "run_ase_core", calculate)
    config = {"configurable": {"thread_id": "one"}}
    database = str(tmp_path / "checkpoints.sqlite")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        graph = graph_for(mode, saver)
        with execution_context(context):
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content="Energy")]}, config
            )
        artifacts = result_artifacts(state, context.task_id)
        assert len(artifacts) == 1
        assert artifacts[0]["tool_call_id"] == "energy"
        assert artifacts[0]["source_path"] == "energy.json"
        content = Path(artifacts[0]["path"]).read_bytes()
        assert context.controller.calls == [("alice", "energy")]
        assert len(invoked) == 1

    (context.workspace / "energy.json").write_text("overwritten")
    async with AsyncSqliteSaver.from_conn_string(database) as saver:
        restored = graph_for(mode, saver)
        snapshot = await restored.aget_state(config)
        assert result_artifacts(snapshot.values, context.task_id) == artifacts
        assert Path(artifacts[0]["path"]).read_bytes() == content
        # Reconstruct the graph and replay the same native call from its journal.
        with execution_context(context):
            replay = await restored.ainvoke(
                {"messages": [HumanMessage(content="Replay")]},
                {"configurable": {"thread_id": "replay"}},
            )
        assert len(invoked) == 1
        assert result_artifacts(replay, context.task_id) == artifacts
        assert (context.workspace / "energy.json").read_text() == "overwritten"


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["registry", "attached", "delegated"])
async def test_journal_failure_stops_all_native_paths(tmp_path, mode):
    class Unavailable(Controller):
        async def execute_tool(self, context, call, invoke):
            raise OSError("journal unavailable")

    context = scope(tmp_path, controller=Unavailable())
    seed(context)
    with (
        execution_context(context),
        pytest.raises(ScopedExecutionError, match="journal unavailable"),
    ):
        await graph_for(mode, InMemorySaver()).ainvoke(
            {"messages": [HumanMessage(content="Energy")]},
            {"configurable": {"thread_id": "one"}},
        )
    assert not (context.workspace / "energy.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["registry", "attached", "delegated"])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_scoped_native_approvals(tmp_path, mode, decision):
    context = scope(tmp_path)
    seed(context)
    graph = graph_for(
        mode,
        InMemorySaver(),
        policy={
            "run_ase": {"allowed_decisions": ["approve", "reject"]},
        },
    )
    config = {"configurable": {"thread_id": "one"}}
    with execution_context(context):
        state = await graph.ainvoke(
            {"messages": [HumanMessage(content="Energy")]}, config
        )
        assert state["__interrupt__"]
        assert context.controller.calls == []
        state = await graph.ainvoke(
            Command(resume={"decisions": [{"type": decision}]}), config
        )
    assert bool(context.controller.calls) == (decision == "approve")
    assert (context.workspace / "energy.json").exists() == (decision == "approve")
    assert bool(result_artifacts(state, context.task_id)) == (decision == "approve")


@pytest.mark.asyncio
async def test_worker_error_handler_cannot_swallow_journal_failure(tmp_path):
    from chemgraph.graphs.tool_review import reviewed_tool_node
    from chemgraph.state.state import State
    from langgraph.graph import StateGraph, START, END

    class Unavailable(Controller):
        async def execute_tool(self, context, call, invoke):
            raise OSError("journal unavailable")

    context = scope(tmp_path, controller=Unavailable())
    node = reviewed_tool_node([run_ase], state_schema=State, handle_tool_errors=True)
    builder = StateGraph(State)
    builder.add_node("tools", node)
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    message = call("run_ase", {}, "energy")
    with (
        execution_context(context),
        pytest.raises(ScopedExecutionError, match="journal unavailable"),
    ):
        await builder.compile().ainvoke({"messages": [message]})


@pytest.mark.asyncio
async def test_parallel_workers_keep_separate_artifact_descriptors(tmp_path):
    from chemgraph.graphs.single_agent import construct_single_agent_graph

    context = scope(tmp_path)
    seed(context)
    workers = []
    for name in ("first", "second"):
        model = _ScriptedChatModel(
            responses=[
                call(
                    "run_ase",
                    {
                        "params": {
                            "input_structure_file": "h2.xyz",
                            "output_results_file": "energy.json",
                            "calculator": {"calculator_type": "emt"},
                            "driver": "energy",
                        }
                    },
                    name,
                ),
                AIMessage(content=f"{name} done"),
            ]
        )
        workers.append(
            {
                "name": name,
                "description": name,
                "runnable": construct_single_agent_graph(
                    model, tools=[run_ase], checkpointer=None
                ),
            }
        )
    model = _ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "id": f"task-{name}",
                        "args": {
                            "subagent_type": name,
                            "description": "Energy",
                        },
                    }
                    for name in ("first", "second")
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    graph = construct_main_agent_graph(
        model,
        subagents=workers,
        discover_skills=False,
        interrupt_on=None,
        checkpointer=InMemorySaver(),
    )
    with execution_context(context):
        state = await asyncio.wait_for(
            graph.ainvoke(
                {"messages": [HumanMessage(content="Run both")]},
                {"configurable": {"thread_id": "parallel"}},
            ),
            timeout=10,
        )
    messages = [
        message
        for message in state["messages"]
        if message.type == "tool" and message.tool_call_id.startswith("task-")
    ]
    assert len(messages) == 2
    for message in messages:
        artifacts = result_artifacts({"messages": [message]}, context.task_id)
        assert len(artifacts) == 1
        assert artifacts[0]["tool_call_id"] == message.tool_call_id.removeprefix(
            "task-"
        )
    assert len(result_artifacts(state, context.task_id)) == 2


@pytest.mark.parametrize("mode", ["registry", "attached", "delegated"])
def test_scoped_synchronous_graph_is_rejected(tmp_path, mode):
    context = scope(tmp_path)
    seed(context)
    with (
        execution_context(context),
        pytest.raises(ScopedExecutionError, match="async graph API"),
    ):
        graph_for(mode, InMemorySaver()).invoke(
            {"messages": [HumanMessage(content="Energy")]},
            {"configurable": {"thread_id": "sync"}},
        )
    assert context.controller.calls == []
    assert not (context.workspace / "energy.json").exists()


@pytest.mark.asyncio
async def test_capture_failure_is_not_committed_as_success(tmp_path, monkeypatch):
    from chemgraph.execution import scoped

    context = scope(tmp_path)
    seed(context)

    def fail(*args):
        raise OSError("snapshot unavailable")

    monkeypatch.setattr(scoped, "_capture", fail)
    with (
        execution_context(context),
        pytest.raises(ScopedExecutionError, match="snapshot unavailable"),
    ):
        await graph_for("delegated", InMemorySaver()).ainvoke(
            {"messages": [HumanMessage(content="Energy")]},
            {"configurable": {"thread_id": "capture"}},
        )
    assert (context.workspace / "energy.json").exists()
    assert context.controller.saved == {}


@pytest.mark.asyncio
async def test_clarification_interrupt_bypasses_journal(tmp_path):
    from chemgraph.tools.generic_tools import ask_human

    context = scope(tmp_path)
    model = _ScriptedChatModel(
        responses=[
            call("ask_human", {"question": "Which structure?"}, "question"),
            AIMessage(content="Understood"),
        ]
    )
    graph = construct_main_agent_graph(
        model,
        main_tools=[ask_human],
        tool_registry=ToolRegistry([]),
        discover_skills=False,
        interrupt_on=None,
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "question"}}
    with execution_context(context):
        state = await graph.ainvoke(
            {"messages": [HumanMessage(content="Calculate")]}, config
        )
        assert state["__interrupt__"]
        assert context.controller.calls == []
        state = await graph.ainvoke(Command(resume="H2"), config)
    assert not state.get("__interrupt__")
    assert context.controller.calls == []
