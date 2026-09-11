"""Regressions for workspace namespaces, approvals, and async checkpoints."""

import json
import sys
from types import SimpleNamespace
from typing import TypedDict

import pytest
from deepagents.backends import LocalShellBackend
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Interrupt, interrupt

from chemgraph.agent.interrupts import collect_pending_interrupts, normalize_interrupts
from chemgraph.agent.llm_agent import ChemGraph, HumanInputRequired
from chemgraph.cli import commands
from chemgraph.graphs.deep_agent import _normalize_backend
from chemgraph.models.endpoints import PreparedModel
from tests.test_deep_agent import _shell_command
from tests.test_main_agent import _ScriptedChatModel


def _agent(monkeypatch, tmp_path, *, responses=None, workflow=None, **kwargs):
    model = _ScriptedChatModel(responses=responses or [AIMessage(content="Done")])
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        lambda **_: (
            model,
            PreparedModel(
                endpoint_name="test",
                protocol="openai_compatible",
                client_kwargs={},
            ),
        ),
    )
    if workflow is not None:
        monkeypatch.setattr(
            "chemgraph.agent.llm_agent.construct_deep_agent_graph",
            lambda *_, **__: workflow,
        )
    return ChemGraph(
        workflow_type="deep_agent",
        log_dir=str(tmp_path / "logs"),
        memory_db_path=str(tmp_path / "memory.db"),
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_workspace_has_one_file_namespace(tmp_path, asynchronous):
    (tmp_path / "a.py").write_text("# TODO a\n")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub/b.py").write_text("# TODO b\n")
    backend = _normalize_backend(LocalShellBackend(root_dir=tmp_path, env={}))
    results = {}

    class Files(TypedDict):
        files: dict

    async def inspect_workspace(state):
        for path in (None, "/", "/workspace/"):
            glob = (
                await backend.aglob("**/*.py", path)
                if asynchronous
                else backend.glob("**/*.py", path)
            )
            grep = (
                await backend.agrep("TODO", path)
                if asynchronous
                else backend.grep("TODO", path)
            )
            results[path] = (glob, grep)
        results["ls"] = await backend.als("/") if asynchronous else backend.ls("/")
        if asynchronous:
            await backend.awrite("/scratch.txt", "checkpoint only")
            await backend.awrite("/workspace/written.txt", "host file")
        else:
            backend.write("/scratch.txt", "checkpoint only")
            backend.write("/workspace/written.txt", "host file")
        command = _shell_command(sys.executable, "-c", "print('executed')")
        results["shell"] = (
            await backend.aexecute(command)
            if asynchronous
            else backend.execute(command)
        )
        return {}

    builder = StateGraph(Files)
    builder.add_node("inspect", inspect_workspace)
    builder.add_edge(START, "inspect")
    builder.add_edge("inspect", END)
    state = await builder.compile().ainvoke({"files": {}})
    for path in (None, "/", "/workspace/"):
        glob, grep = results[path]
        assert sorted(item["path"] for item in glob.matches) == [
            "/workspace/a.py",
            "/workspace/sub/b.py",
        ]
        assert sorted(item["path"] for item in grep.matches) == [
            "/workspace/a.py",
            "/workspace/sub/b.py",
        ]
        assert not glob.truncated and not grep.truncated
    assert [item["path"] for item in results["ls"].entries] == ["/workspace/"]
    assert "/scratch.txt" in state["files"]
    assert not (tmp_path / "scratch.txt").exists()
    assert (tmp_path / "written.txt").read_text() == "host file"
    assert results["shell"].exit_code == 0
    assert "executed" in results["shell"].output


def _approval():
    return {
        "action_requests": [{"name": "write_file", "args": {}}],
        "review_configs": [
            {"action_name": "write_file", "allowed_decisions": ["approve", "reject"]}
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("via_cli", [False, True])
async def test_twelve_file_approvals_complete(monkeypatch, tmp_path, via_cli):
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "args": {"file_path": f"/workspace/{i}.txt", "content": str(i)},
                    "id": f"write-{i}",
                    "type": "tool_call",
                }
            ],
        )
        for i in range(12)
    ] + [AIMessage(content="Done")]
    decisions = []

    def approve(payload):
        decisions.append(payload)
        return {"decisions": [{"type": "approve"}]}

    agent = _agent(
        monkeypatch,
        tmp_path,
        responses=responses,
        deepagent_backend=LocalShellBackend(root_dir=tmp_path, env={}),
        human_input_handler=None
        if via_cli
        else lambda question, payload: approve(payload),
    )
    if via_cli:
        monkeypatch.setattr(commands, "_prompt_for_interrupt", approve)
        monkeypatch.setattr(commands.time, "sleep", lambda _: None)
        result = commands.run_query(agent, "Write twelve files", thread_id=37)
    else:
        result = await agent.run("Write twelve files", config={"thread_id": 37})
    assert result.content == "Done"
    assert len(decisions) == 12
    assert all((tmp_path / f"{i}.txt").read_text() == str(i) for i in range(12))
    assert len(agent.session_store.get_session(agent.session_id).messages) == 14


@pytest.mark.asyncio
@pytest.mark.parametrize("return_option", ["state", "last_message"])
async def test_async_saver_recovers_approval_and_logs(
    monkeypatch, tmp_path, return_option
):
    builder = StateGraph(MessagesState)

    def ask(state):
        decision = interrupt(_approval())
        assert decision == {"decisions": [{"type": "approve"}]}
        return {"messages": [AIMessage(content="Approved")]}

    builder.add_node("ask", ask)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    config = {"configurable": {"thread_id": "async-saver"}}
    async with AsyncSqliteSaver.from_conn_string(
        str(tmp_path / "checkpoints.db")
    ) as saver:
        workflow = builder.compile(checkpointer=saver)
        agent = _agent(
            monkeypatch, tmp_path, workflow=workflow, return_option=return_option
        )
        with pytest.raises(HumanInputRequired) as exc:
            await agent.run("Approve", config=config)
        assert exc.value.interrupts[0].id
        final = await workflow.ainvoke(
            Command(resume={"decisions": [{"type": "approve"}]}), config
        )
        result = await agent.afinalize_completed_run(final, config, "Approve")
        assert (
            result["messages"][-1]["content"] == "Approved"
            if return_option == "state"
            else result.content == "Approved"
        )
    logs = list((tmp_path / "logs").glob("*.json"))
    assert len(logs) == 2
    assert all(
        json.loads(log.read_text())["thread_id"] == "async-saver" for log in logs
    )
    assert len(agent.session_store.get_session(agent.session_id).messages) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("handler", [False, True])
async def test_bare_interrupt_uses_checkpoint_request(monkeypatch, tmp_path, handler):
    class Workflow:
        pending = True
        state = {"messages": [HumanMessage(content="Continue")]}

        async def astream(self, value, **kwargs):
            if not isinstance(value, Command):
                raise GraphInterrupt()
            self.pending = False
            self.state = {"messages": [AIMessage(content="Done")]}
            yield self.state

        async def aget_state(self, config):
            pending = (
                (Interrupt(value={"question": "Continue?"}, id="real-id"),)
                if self.pending
                else ()
            )
            return SimpleNamespace(values=self.state, interrupts=pending, tasks=())

    agent = _agent(
        monkeypatch,
        tmp_path,
        workflow=Workflow(),
        human_input_handler=(lambda _: "yes") if handler else None,
    )
    if handler:
        assert (await agent.run("Continue")).content == "Done"
    else:
        with pytest.raises(HumanInputRequired) as exc:
            await agent.run("Continue")
        assert [item.id for item in exc.value.interrupts] == ["real-id"]


def test_anonymous_interrupts_are_not_collapsed_or_counted_twice():
    raw = [Interrupt(value={"a": 1, "b": 2}), Interrupt(value={"b": 2, "a": 1})]
    normalized = normalize_interrupts(raw)
    assert [item.id for item in normalized] == ["", ""]
    snapshot = SimpleNamespace(interrupts=raw, tasks=[SimpleNamespace(interrupts=raw)])
    assert collect_pending_interrupts(normalized, snapshot) == tuple(normalized)
    assert len(collect_pending_interrupts(normalized)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("approval", [False, True])
async def test_concurrent_interrupts_limit_only_questions(
    monkeypatch, tmp_path, approval
):
    builder = StateGraph(MessagesState)

    def ask(state):
        interrupt(_approval() if approval else {"question": "Continue?"})
        return {"messages": [AIMessage(content="Done")]}

    for i in range(11):
        builder.add_node(str(i), ask)
        builder.add_edge(START, str(i))
        builder.add_edge(str(i), END)
    agent = _agent(
        monkeypatch,
        tmp_path,
        workflow=builder.compile(checkpointer=InMemorySaver()),
        human_input_handler=lambda question, payload: (
            {"decisions": [{"type": "approve"}]} if approval else "yes"
        ),
    )
    if approval:
        assert (await agent.run("Continue")).content == "Done"
    else:
        with pytest.raises(RuntimeError, match="maximum of 10"):
            await agent.run("Continue")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_finalization_scopes_idless_messages_to_thread(
    monkeypatch, tmp_path, asynchronous
):
    state = {
        "messages": [
            {"type": "human", "content": "Repeated"},
            {"type": "ai", "content": "Same reply"},
        ]
    }

    class Workflow:
        def get_state(self, config):
            return SimpleNamespace(values=state)

        async def aget_state(self, config):
            return self.get_state(config)

    agent = _agent(monkeypatch, tmp_path, workflow=Workflow(), return_option="state")
    agent._ensure_session("Repeated")
    for thread in (7, "7", "new-thread", "new-thread"):
        config = {"configurable": {"thread_id": thread}}
        if asynchronous:
            await agent.afinalize_completed_run(state, config, "Repeated")
        else:
            agent.finalize_completed_run(state, config, "Repeated")
    session = agent.session_store.get_session(agent.session_id)
    assert [msg.content for msg in session.messages] == ["Repeated", "Same reply"] * 2
    assert session.query_count == 2


def test_registry_worker_delegates_using_canonical_name():
    from chemgraph.graphs.main_agent import (
        DEFAULT_MAIN_AGENT_PROMPT,
        construct_main_agent_graph,
    )
    from chemgraph.registry import AgentRegistry

    worker = AgentRegistry().as_subagent(
        "deepagent",
        llm=_ScriptedChatModel(responses=[AIMessage(content="Workspace reviewed")]),
    )
    assert worker["name"] == "deep_agent"
    assert worker["name"] in DEFAULT_MAIN_AGENT_PROMPT
    model = _ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "subagent_type": worker["name"],
                            "description": "Review workspace",
                        },
                        "id": "delegate",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    workflow = construct_main_agent_graph(model, subagents=[worker])
    result = workflow.invoke(
        {"messages": [HumanMessage(content="Review workspace")]},
        {"configurable": {"thread_id": "registry"}},
    )
    assert any(
        msg.type == "tool" and "Workspace reviewed" in str(msg.content)
        for msg in result["messages"]
    )


@pytest.mark.asyncio
async def test_placeholder_concurrent_requests_are_rejected_before_handler(
    monkeypatch, tmp_path
):
    class Workflow:
        async def astream(self, value, **kwargs):
            yield {
                "messages": [],
                "__interrupt__": [
                    Interrupt(value=_approval()),
                    Interrupt(value=_approval()),
                ],
            }

        async def aget_state(self, config):
            return SimpleNamespace(values={"messages": []}, interrupts=(), tasks=())

    agent = _agent(
        monkeypatch,
        tmp_path,
        workflow=Workflow(),
        human_input_handler=lambda *_: pytest.fail(
            "must reject ambiguous requests before prompting"
        ),
    )
    with pytest.raises(RuntimeError, match="stable IDs"):
        await agent.run("Continue")


@pytest.mark.asyncio
async def test_snapshot_failure_preserves_resumable_stream_interrupt(
    monkeypatch, tmp_path
):
    class Workflow:
        async def astream(self, value, **kwargs):
            yield {
                "messages": [],
                "__interrupt__": [Interrupt(value=_approval(), id="pending")],
            }

        async def aget_state(self, config):
            raise RuntimeError("checkpoint read unavailable")

    agent = _agent(monkeypatch, tmp_path, workflow=Workflow())
    with pytest.raises(HumanInputRequired) as exc:
        await agent.run("Continue")
    assert exc.value.interrupts[0].id == "pending"
