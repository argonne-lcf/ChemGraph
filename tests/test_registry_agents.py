"""Lazy worker discovery, delegated reviews, and durable private state."""

from dataclasses import replace
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from chemgraph.agent.main_session import MainAgentSession
from chemgraph.cli.checkpoint_runtime import CheckpointRuntime
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.graphs.single_agent import construct_single_agent_graph
from chemgraph.registry.agents import AgentRegistry, AgentSpec
from chemgraph.registry.tools import ToolRegistry
from tests.test_main_agent import _ScriptedChatModel


EXECUTIONS = []
BUILDS = []


@tool
def run_ase(value: int = 1) -> str:
    """Hermetic calculation stub."""
    EXECUTIONS.append(value)
    return f"energy={value} eV"


def build_worker(llm, *, interrupt_on=None, checkpointer=None, **kwargs):
    BUILDS.append("worker")
    return construct_single_agent_graph(
        llm, tools=[run_ase], interrupt_on=interrupt_on,
        checkpointer=checkpointer, **kwargs,
    )


SPEC = AgentSpec("worker", "Hermetic chemistry worker", "tests.test_registry_agents:build_worker",
                 aliases=("chemistry",), default_tool_names=("run_ase",))


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{name}-{uuid4().hex}"}])


def make_graph(responses, **kwargs):
    return construct_main_agent_graph(
        _ScriptedChatModel(responses=responses), agent_registry=AgentRegistry([SPEC]),
        tool_registry=ToolRegistry([]), discover_skills=False, **kwargs,
    )


@pytest.fixture(autouse=True)
def reset():
    EXECUTIONS.clear()
    BUILDS.clear()


@pytest.mark.asyncio
async def test_discovery_is_metadata_only_and_loading_normalizes_aliases():
    graph = make_graph([call("search_agents", query="chemistry"),
                        call("load_agents", names=["chemistry", "worker"]),
                        AIMessage(content="ready")])
    assert not BUILDS
    session = MainAgentSession(graph)
    result = await session.run("Find a specialist")
    assert BUILDS == ["worker"]
    assert result.state["active_registry_agents"] == []
    assert any('"available": true' in message.content for message in
               (await graph.aget_state(session.config)).values["messages"]
               if isinstance(message, ToolMessage) and message.name == "search_agents")


@pytest.mark.asyncio
@pytest.mark.parametrize("worker", ["worker", "general-purpose"])
async def test_unloaded_and_implicit_workers_cannot_execute(worker):
    graph = make_graph([call("task", subagent_type=worker, description="calculate"),
                        AIMessage(content="not delegated")])
    session = MainAgentSession(graph)
    result = await session.run("Calculate")
    assert result.assistant_response == "not delegated"
    assert BUILDS == [] and EXECUTIONS == []


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_delegated_review_survives_sqlite_restart(tmp_path, decision):
    database = str(tmp_path / "checkpoints.db")
    for restart in (False, True):
        runtime = CheckpointRuntime()
        try:
            responses = ([AIMessage(content="worker done"), AIMessage(content="main done")]
                         if restart else [call("load_agents", names=["chemistry"]),
                         call("task", subagent_type="chemistry", description="calculate"), call("run_ase")])
            graph = make_graph(responses, checkpointer=runtime.open_sqlite(database))
            session = MainAgentSession(graph, thread_id="restart", configuration_id="review-v1")
            if not restart:
                pending = runtime.run(lambda: session.run("Calculate"))
                assert pending.status == "waiting_for_user"
                assert pending.state["active_registry_agents"] == ["worker"]
                assert EXECUTIONS == []
            else:
                assert runtime.run(session.restore).status == "waiting_for_user"
                result = runtime.run(lambda: session.resume({"decisions": [{"type": decision}]}))
                assert result.assistant_response == "main done"
                assert result.state["active_registry_agents"] == []
                assert runtime.run(session.restore).status == "completed"
                assert EXECUTIONS == ([1] if decision == "approve" else [])
        finally:
            runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("choices,expected", [(["approve", "reject"], [1]), (["reject", "reject"], [])])
async def test_legacy_review_batches_skip_rejected_handlers(choices, expected):
    batch = AIMessage(content="", tool_calls=[
        {"name": "run_ase", "args": {"value": value}, "id": str(value)} for value in (1, 2)
    ])
    graph = make_graph([call("load_agents", names=["worker"]),
                        call("task", subagent_type="worker", description="calculate"),
                        batch, AIMessage(content="worker done"), AIMessage(content="done")])
    session = MainAgentSession(graph)
    assert (await session.run("Calculate")).status == "waiting_for_user"
    assert EXECUTIONS == []
    assert (await session.resume({"decisions": [{"type": value} for value in choices]})).status == "completed"
    assert EXECUTIONS == expected


@pytest.mark.asyncio
async def test_failed_load_preserves_previous_selection_and_worker_policy_cannot_disable_review():
    graph = make_graph([call("load_agents", names=["worker"]),
                        call("load_agents", names=["missing"]),
                        call("task", subagent_type="worker", description="calculate"), call("run_ase"),
                        AIMessage(content="worker done"), AIMessage(content="done")],
                       agent_options={"worker": {"interrupt_on": {"run_ase": False}}})
    session = MainAgentSession(graph)
    pending = await session.run("Calculate")
    assert pending.state["active_registry_agents"] == ["worker"]
    assert EXECUTIONS == []
    await session.resume({"decisions": [{"type": "reject"}]})
    assert EXECUTIONS == []


@pytest.mark.asyncio
async def test_raw_recursion_limit_round_trips():
    graph = make_graph([AIMessage(content="done")], recursion_limit=17)
    original = MainAgentSession(graph, configuration_id="raw-v1", thread_id="recursion")
    await original.run("hello")
    assert original.session_metadata.graph_config.recursion_limit == 17
    restored = MainAgentSession(graph, thread_id=original.thread_id, session_metadata=original.session_metadata)
    await restored.restore()
    assert restored.config["recursion_limit"] == 17


@pytest.mark.asyncio
async def test_worker_selection_clears_before_next_turn():
    graph = make_graph([call("load_agents", names=["worker"]), AIMessage(content="ready"),
                        call("task", subagent_type="worker", description="calculate"),
                        AIMessage(content="load again first")])
    session = MainAgentSession(graph)
    assert (await session.run("Load chemistry")).state["active_registry_agents"] == []
    result = await session.run("Calculate next")
    assert result.assistant_response == "load again first"
    assert EXECUTIONS == [] and BUILDS == ["worker"]


@pytest.mark.asyncio
async def test_worker_selection_survives_failed_model_and_retry():
    graph = make_graph([call("load_agents", names=["worker"]), RuntimeError("temporary model failure"),
                        call("task", subagent_type="worker", description="calculate"),
                        AIMessage(content="worker done"), AIMessage(content="done")])
    session = MainAgentSession(graph)
    with pytest.raises(RuntimeError, match="temporary model failure"):
        await session.run("Calculate")
    assert (await graph.aget_state(session.config)).values["active_registry_agents"] == ["worker"]
    result = await session.retry()
    assert result.assistant_response == "done"
    assert result.state["active_registry_agents"] == []
    assert BUILDS == ["worker"]


def broken_worker(_llm, **_kwargs):
    raise RuntimeError("specialist initialization failed")


@pytest.mark.asyncio
async def test_constructor_failure_returns_tool_error_and_keeps_selection():
    broken = replace(SPEC, name="broken", aliases=(), import_path="tests.test_registry_agents:broken_worker")
    model = _ScriptedChatModel(responses=[call("load_agents", names=["worker"]),
        call("load_agents", names=["broken"]),
        call("task", subagent_type="worker", description="calculate"),
        AIMessage(content="worker done"), AIMessage(content="done")])
    graph = construct_main_agent_graph(model, agent_registry=AgentRegistry([SPEC, broken]))
    session = MainAgentSession(graph)
    result = await session.run("Calculate")
    assert result.assistant_response == "done"
    assert any(isinstance(message, ToolMessage) and message.status == "error"
               and "specialist initialization failed" in message.content
               for message in (await graph.aget_state(session.config)).values["messages"])
    assert BUILDS == ["worker"]


def test_reviews_do_not_duplicate_planner_results():
    from chemgraph.graphs.tool_review import reviewed_tool_node
    from chemgraph.state.graspa_state import PlannerState
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(PlannerState)
    builder.add_node("tools", reviewed_tool_node([run_ase], {"run_ase": False}, state_schema=PlannerState))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    result = builder.compile().invoke({"messages": [call("run_ase")], "executor_results": ["existing"]})
    assert result["executor_results"] == ["existing"]
    assert EXECUTIONS == [1]


def test_empty_worker_catalog_disables_discovery_and_delegation():
    model = _ScriptedChatModel(responses=[AIMessage(content="done")])
    graph = construct_main_agent_graph(model, agent_registry=AgentRegistry([]))
    graph.invoke({"messages": [HumanMessage(content="hello")]}, {"configurable": {"thread_id": "empty"}})
    assert not {"search_agents", "load_agents", "task"} & {item.name for item in model.bound_tools}


def test_long_conversation_checkpoint_size_tracks_standalone_deepagent():
    from chemgraph.graphs.deep_agent import construct_deep_agent_graph

    def message_bytes(saver):
        blobs = sum(len(value[1]) for key, value in saver.blobs.items() if key[2] == "messages")
        writes = sum(len(value[2][1]) for batch in saver.writes.values() for value in batch.values()
                     if value[1] == "messages")
        return blobs + writes

    sizes = []
    for constructor in (construct_deep_agent_graph, construct_main_agent_graph):
        model = _ScriptedChatModel(responses=[AIMessage(content="result " * 200) for _ in range(40)])
        options = {"agent_registry": AgentRegistry([])} if constructor is construct_main_agent_graph else {}
        graph = constructor(model, tool_registry=ToolRegistry([]),
                            discover_skills=False, **options)
        for turn in range(40):
            graph.invoke({"messages": [HumanMessage(content=f"{turn}: " + "query " * 200)]},
                         {"configurable": {"thread_id": "long-conversation"}})
        sizes.append(message_bytes(graph.checkpointer))
    assert sizes[1] <= sizes[0] * 1.5


INSPECTIONS = []


def inspect_worker(_llm, **_kwargs):
    from langchain_core.runnables import RunnableLambda

    def inspect(state):
        INSPECTIONS.append(state)
        return {"messages": [AIMessage(content="inspected")], "files": {
            "/result.txt": {"content": ["worker result"], "created_at": "2026-10-01T00:00:00Z",
                            "modified_at": "2026-10-01T00:00:00Z"},
        }}
    return RunnableLambda(inspect)


@pytest.mark.asyncio
async def test_registry_worker_keeps_private_state_and_exchanges_files():
    INSPECTIONS.clear()
    spec = replace(SPEC, import_path="tests.test_registry_agents:inspect_worker")
    tools = ToolRegistry([ToolRegistry().get_spec("calculator")])
    # Make tool selection observable in the parent without executing the calculator.
    graph = construct_main_agent_graph(_ScriptedChatModel(responses=[
        call("load_tools", names=["calculator"]), call("load_agents", names=["worker"]),
        call("task", subagent_type="worker", description="inspect"), AIMessage(content="done"),
    ]), agent_registry=AgentRegistry([spec]), tool_registry=tools)
    session = MainAgentSession(graph)
    assert (await session.run("Inspect")).assistant_response == "done"
    assert len(INSPECTIONS) == 1
    assert not {"active_registry_agents", "active_registry_tools", "skills_metadata", "skills_load_errors"} & INSPECTIONS[0].keys()
    assert (await graph.aget_state(session.config)).values["files"]["/result.txt"]["content"] == ["worker result"]
