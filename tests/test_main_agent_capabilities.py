"""Direct main-agent work, configured delegation, and durable approvals."""

from typing import Any, NotRequired

import pytest
from deepagents.backends import LocalShellBackend
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph

from chemgraph.agent.main_session import MainAgentSession
from chemgraph.cli.checkpoint_runtime import CheckpointRuntime
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.memory.schemas import MainAgentGraphConfig, MainAgentSessionMetadata
from chemgraph.memory.store import SessionStore
from chemgraph.registry.tools import ToolRegistry
from tests.test_main_agent import (
    _FileState, _ScriptedChatModel, _answering_subgraph, _subagent,
)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def graph(model, **kwargs):
    kwargs.setdefault("subagents", [_subagent(_answering_subgraph("worker result"))])
    return construct_main_agent_graph(model, **kwargs)


def test_main_agent_owns_runtime_and_default_workers(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The main agent must not be built through the DeepAgent factory")

    monkeypatch.setattr("chemgraph.graphs.deep_agent.create_deep_agent", forbidden)
    monkeypatch.setattr("chemgraph.graphs.main_agent.construct_deep_agent_graph", forbidden)
    model = _ScriptedChatModel(responses=[AIMessage(content="done")])
    workflow = construct_main_agent_graph(model)
    workflow.invoke(
        {"messages": [HumanMessage(content="hello")]},
        {"configurable": {"thread_id": "own-runtime"}},
    )
    tools = {item.name: item for item in model.bound_tools}
    assert {"write_file", "edit_file", "read_file", "search_tools", "load_tools"} <= tools.keys()
    assert "execute" not in tools
    assert "chemgraph" in tools["task"].description
    assert "deepagent" not in tools["task"].description
    assert "general-purpose" not in tools["task"].description
    state = workflow.get_state({"configurable": {"thread_id": "own-runtime"}}).values
    assert any(skill["name"] == "chemgraph" for skill in state["skills_metadata"])


class PrivateState(_FileState):
    active_registry_tools: NotRequired[list[str]]
    skills_metadata: NotRequired[list[Any]]


@pytest.mark.asyncio
async def test_direct_registry_work_and_delegation_keep_private_state_isolated():
    seen = []

    @tool
    def double(value: int) -> int:
        """Double a number."""
        seen.append(value)
        return value * 2

    def worker(state):
        assert "active_registry_tools" not in state
        assert "skills_metadata" not in state
        return {
            "messages": [AIMessage(content="checked")],
            "active_registry_tools": ["unavailable"],
            "skills_metadata": [{"name": "unwanted"}],
        }

    builder = StateGraph(PrivateState)
    builder.add_node("worker", worker)
    builder.add_edge(START, "worker")
    builder.add_edge("worker", END)
    model = _ScriptedChatModel(responses=[
        call("load_tools", names=["double"]), call("double", value=2),
        call("task", subagent_type="worker", description="check result"),
        call("double", value=3), AIMessage(content="4 and 6, checked"),
    ])
    registry = ToolRegistry([])
    registry.register(double)
    workflow = graph(model, tool_registry=registry,
                     subagents=[_subagent(builder.compile(checkpointer=None))])
    result = await MainAgentSession(workflow, thread_id="private-state").run("Calculate and check")
    assert result.assistant_response == "4 and 6, checked"
    assert seen == [2, 3]
    assert result.state["active_registry_tools"] == []
    assert "unwanted" not in {skill["name"] for skill in result.state["skills_metadata"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_main_agent_reads_skills_and_reviews_workspace_edits(tmp_path, decision):
    skill = tmp_path / ".agents/skills/example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: example\ndescription: Test skill\n---\nUse EMT.\n")
    target = tmp_path / "input.txt"
    target.write_text("before\n")
    model = _ScriptedChatModel(responses=[
        call("read_file", file_path="/workspace/.agents/skills/example/SKILL.md"),
        call("edit_file", file_path="/workspace/input.txt", old_string="before", new_string="after"),
        AIMessage(content="handled"),
    ])
    workflow = graph(model, backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True, env={}),
                     user_skills_dir=str(tmp_path / "personal"))
    session = MainAgentSession(workflow, thread_id="workspace-review")
    pending = await session.run("Read the skill and edit the input")
    assert pending.status == "waiting_for_user"
    assert target.read_text() == "before\n"
    assert "example" in {skill["name"] for skill in pending.state["skills_metadata"]}
    snapshot = await workflow.aget_state(session.config)
    assert any("Use EMT." in message.content for message in snapshot.values["messages"]
               if isinstance(message, ToolMessage) and message.name == "read_file")
    result = await session.resume({"decisions": [{"type": decision}]})
    assert result.assistant_response == "handled"
    assert target.read_text() == ("after\n" if decision == "approve" else "before\n")


def test_registry_approval_survives_restart(tmp_path):
    checkpoint_db = tmp_path / "checkpoints.db"
    store = SessionStore(str(tmp_path / "sessions.db"))
    metadata = MainAgentSessionMetadata(
        graph_config=MainAgentGraphConfig(model_name="scripted", graph_schema_version=2,
                                            configuration_id="ase-stub-v1", topology_fingerprint="ase-stub-v1"),
        checkpoint_backend="AsyncSqliteSaver", checkpoint_db=str(checkpoint_db),
    )
    executions = []

    @tool
    def run_ase() -> str:
        """Hermetic calculation stub."""
        executions.append("ran")
        return "energy=-1 eV"

    registry = ToolRegistry([])
    registry.register(run_ase)
    for restart in (False, True):
        runtime = CheckpointRuntime()
        try:
            saver = runtime.open_sqlite(str(checkpoint_db))
            responses = [AIMessage(content="energy=-1 eV")] if restart else [
                call("load_tools", names=["run_ase"]), call("run_ase"),
            ]
            workflow = graph(_ScriptedChatModel(responses=responses),
                             tool_registry=registry, checkpointer=saver)
            session = MainAgentSession(workflow, thread_id="restart", session_store=store,
                                       session_metadata=metadata)
            if not restart:
                result = runtime.run(lambda: session.run("Calculate"))
                assert result.status == "waiting_for_user" and not executions
            else:
                assert runtime.run(session.restore).status == "waiting_for_user"
                result = runtime.run(lambda: session.resume({"decisions": [{"type": "approve"}]}))
                assert result.assistant_response == "energy=-1 eV"
                assert executions == ["ran"]
        finally:
            runtime.close()


@pytest.mark.asyncio
async def test_direct_return_uses_current_tool_result_and_restores():
    @tool(return_direct=True)
    def answer() -> str:
        """Return the result directly."""
        return "current result"

    registry = ToolRegistry([])
    registry.register(answer)
    workflow = graph(_ScriptedChatModel(responses=[
        AIMessage(content="previous answer"), call("load_tools", names=["answer"]), call("answer"),
    ]), tool_registry=registry)
    session = MainAgentSession(workflow, thread_id="direct-return")
    await session.run("First question")
    assert (await session.run("Second question")).assistant_response == "current result"
    assert (await session.restore()).assistant_response == "current result"
    snapshot = await workflow.aget_state(session.config)
    assert isinstance(snapshot.values["messages"][-1], ToolMessage)


@pytest.mark.asyncio
async def test_main_agent_shell_waits_for_approval(tmp_path):
    target = tmp_path / "shell-result.txt"
    workflow = graph(_ScriptedChatModel(responses=[
        call("execute", command="printf done > shell-result.txt"), AIMessage(content="done"),
    ]), backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True, env={}))
    session = MainAgentSession(workflow, thread_id="shell")
    assert (await session.run("Run the command")).status == "waiting_for_user"
    assert not target.exists()
    await session.resume({"decisions": [{"type": "approve"}]})
    assert target.read_text() == "done"


@pytest.mark.asyncio
async def test_summarization_keeps_readable_history_and_private_state(monkeypatch, tmp_path):
    from deepagents.middleware.summarization import SummarizationMiddleware

    summaries = _ScriptedChatModel(responses=[AIMessage(content="Summary of earlier requests.")] * 10)
    monkeypatch.setattr("chemgraph.graphs.main_agent.create_summarization_middleware",
                        lambda model, backend: SummarizationMiddleware(
                            summaries, backend=backend, trigger=("messages", 4), keep=("messages", 2),
                        ))
    model = _ScriptedChatModel(responses=[AIMessage(content="first"), AIMessage(content="second"),
                                         AIMessage(content="third")])
    workflow = graph(model)
    store = SessionStore(str(tmp_path / "sessions.db"))
    session = MainAgentSession(workflow, session_store=store, configuration_id="summary-test-v1")
    for question in ("question one", "question two", "question three"):
        await session.run(question)
    state = (await workflow.aget_state(session.config)).values
    assert state.get("_summarization_event")
    assert [message.content for message in state["messages"] if isinstance(message, HumanMessage)] == [
        "question one", "question two", "question three",
    ]
    saved = store.get_session(session.thread_id)
    assert len(saved.messages) == 6
    assert saved.query_count == 3
    restored = MainAgentSession(workflow, thread_id=session.thread_id, session_store=store,
                                configuration_id="summary-test-v1")
    assert (await restored.restore()).assistant_response == "third"
    assert restored.session_usage["call_count"] == session.session_usage["call_count"]
