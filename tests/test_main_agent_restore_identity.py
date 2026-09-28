"""Reject incompatible checkpoints before invoking models or reviewed tools."""

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from chemgraph.agent.main_session import IncompatibleCheckpointError, MainAgentSession
from chemgraph.cli.checkpoint_runtime import CheckpointRuntime
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.memory.schemas import MainAgentGraphConfig, MainAgentSessionMetadata
from chemgraph.memory.store import SessionStore
from chemgraph.registry.tools import ToolRegistry
from tests.test_main_agent import _ScriptedChatModel, _answering_subgraph, _subagent, _task_call


def make_graph(responses, **kwargs):
    return construct_main_agent_graph(
        _ScriptedChatModel(responses=responses), tool_registry=ToolRegistry([]),
        discover_skills=False, subagents=kwargs.pop("subagents", [_subagent(_answering_subgraph("done"))]),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["run", "restore"])
@pytest.mark.parametrize("identity", [None, "different"])
async def test_reconstructed_raw_graph_requires_matching_identity(operation, identity, tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    workflow = make_graph([AIMessage(content="first")])
    original = MainAgentSession(workflow, thread_id="raw", session_store=store,
                                configuration_id="original")
    await original.run("hello")
    restored = MainAgentSession(workflow, thread_id="raw", session_store=store,
                                configuration_id=identity)
    with pytest.raises(IncompatibleCheckpointError):
        await (restored.run("must not execute") if operation == "run" else restored.restore())
    assert len((await workflow.aget_state(original.config)).values["messages"]) == 2
    assert not store.get_session_metadata("raw")[1].graph_config.cli_restorable


@pytest.mark.asyncio
async def test_raw_graph_without_identity_only_continues_in_same_instance():
    workflow = make_graph([AIMessage(content="first"), AIMessage(content="second")])
    original = MainAgentSession(workflow, thread_id="raw")
    await original.run("hello")
    assert (await original.restore()).assistant_response == "first"
    await original.run("follow up")
    with pytest.raises(IncompatibleCheckpointError, match="configuration_id"):
        await MainAgentSession(workflow, thread_id="raw").restore()


@pytest.mark.asyncio
@pytest.mark.parametrize("readable", [False, True])
@pytest.mark.parametrize("operation", ["run", "restore"])
async def test_legacy_checkpoints_are_rejected_without_touching_transcripts(tmp_path, readable, operation):
    workflow = make_graph([AIMessage(content="old answer")])
    await workflow.ainvoke({"messages": [HumanMessage(content="old question")]},
                           {"configurable": {"thread_id": "legacy"}})
    store = SessionStore(str(tmp_path / "sessions.db")) if readable else None
    if store:
        store.create_session("legacy", "old", "main_agent", session_metadata=MainAgentSessionMetadata(
            graph_config=MainAgentGraphConfig(model_name="old"),
        ))
    session = MainAgentSession(workflow, thread_id="legacy", session_store=store,
                               configuration_id="new")
    with pytest.raises(IncompatibleCheckpointError, match="Start a new session"):
        await (session.run("new") if operation == "run" else session.restore())
    assert len((await workflow.aget_state(session.config)).values["messages"]) == 2
    if store:
        assert store.get_session("legacy").graph_config.graph_schema_version == 1


@pytest.mark.asyncio
async def test_readable_legacy_record_cannot_be_reused_without_a_checkpoint(tmp_path):
    store = SessionStore(str(tmp_path / "sessions.db"))
    store.create_session("old", "old", "main_agent")
    session = MainAgentSession(make_graph([]), thread_id="old", session_store=store)
    with pytest.raises(IncompatibleCheckpointError, match="Start a new session"):
        await session.run("new")


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("delegated", [False, True])
def test_reviewed_action_restores_in_sqlite_exactly_once(tmp_path, decision, delegated):
    executions = []

    @tool
    def reviewed_action() -> str:
        """Record one approved action."""
        executions.append("ran")
        return "done"

    action = AIMessage(content="", tool_calls=[{
        "name": "reviewed_action", "args": {}, "id": "action-1", "type": "tool_call",
    }])
    database = str(tmp_path / "checkpoints.db")
    for restarting in (False, True):
        runtime = CheckpointRuntime()
        try:
            saver = runtime.open_sqlite(database)
            if delegated:
                child = create_agent(
                    _ScriptedChatModel(responses=[AIMessage(content="worker done")] if restarting
                                       else [action, AIMessage(content="worker done")]),
                    tools=[reviewed_action], checkpointer=None,
                    middleware=[HumanInTheLoopMiddleware(interrupt_on={
                        "reviewed_action": {"allowed_decisions": ["approve", "reject"]},
                    })],
                )
                workflow = make_graph(
                    [AIMessage(content="all done")] if restarting else [
                        AIMessage(content="", tool_calls=[_task_call("delegate")]), AIMessage(content="all done"),
                    ], checkpointer=saver, subagents=[_subagent(child)],
                )
            else:
                workflow = make_graph(
                    [AIMessage(content="all done")] if restarting else [action, AIMessage(content="all done")],
                    checkpointer=saver, main_tools=[reviewed_action],
                    interrupt_on={"reviewed_action": {"allowed_decisions": ["approve", "reject"]}},
                )
            session = MainAgentSession(workflow, thread_id="review", configuration_id="review-policy-v1")
            if not restarting:
                assert runtime.run(lambda: session.run("do it")).status == "waiting_for_user"
                assert executions == []
            else:
                assert runtime.run(session.restore).status == "waiting_for_user"
                result = runtime.run(lambda: session.resume({"decisions": [{"type": decision}]}))
                assert result.status == "completed"
                assert runtime.run(session.restore).status == "completed"
                assert executions == (["ran"] if decision == "approve" else [])
        finally:
            runtime.close()


@pytest.mark.asyncio
async def test_tampered_checkpoint_is_rejected_before_resume():
    from langgraph.types import interrupt
    from langgraph.graph import StateGraph, MessagesState, START, END

    def pause(_state):
        interrupt("continue?")
        return {"messages": [AIMessage(content="done")]}
    builder = StateGraph(MessagesState)
    builder.add_node("pause", pause)
    builder.add_edge(START, "pause")
    builder.add_edge("pause", END)
    workflow = builder.compile(checkpointer=InMemorySaver())
    session = MainAgentSession(workflow, thread_id="pending", configuration_id="policy-v1")
    await session.run("hello")
    # A different writer replaced the checkpoint metadata while this owner paused.
    await workflow.aupdate_state({"configurable": {"thread_id": "pending"},
                                  "metadata": {"chemgraph_topology": "other"}}, {})
    with pytest.raises(IncompatibleCheckpointError):
        await session.resume("yes")
