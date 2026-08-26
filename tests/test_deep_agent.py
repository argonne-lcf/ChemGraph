"""Tests for the reusable standalone Deep Agent workflow."""

import json
import shlex
import sys
from types import SimpleNamespace
from typing import Any

import pytest
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import Field

from chemgraph.agent.llm_agent import ChemGraph, PromptConfig
from chemgraph.cli import commands
from chemgraph.graphs.deep_agent import (
    DEFAULT_DEEPAGENT_INTERRUPT_ON,
    construct_deep_agent_graph,
)
from chemgraph.models.endpoints import PreparedModel
from tests.test_main_agent import _ScriptedChatModel


class _FakeWorkflow:
    def __init__(self):
        self.config = None

    def with_config(self, config):
        self.config = config
        return self


class _RecordingChatModel(_ScriptedChatModel):
    received_messages: list[Any] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received_messages = list(messages)
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def test_constructor_builds_safe_standalone_graph(monkeypatch):
    captured = {}
    workflow = _FakeWorkflow()

    def fake_create_deep_agent(**kwargs):
        captured.update(kwargs)
        return workflow

    monkeypatch.setattr(
        "chemgraph.graphs.deep_agent.create_deep_agent",
        fake_create_deep_agent,
    )

    result = construct_deep_agent_graph(object(), recursion_limit=17)

    assert result is workflow
    assert captured["tools"] == []
    assert isinstance(captured["backend"], StateBackend)
    assert isinstance(captured["checkpointer"], MemorySaver)
    assert captured["interrupt_on"] == DEFAULT_DEEPAGENT_INTERRUPT_ON
    assert captured["interrupt_on"] is not DEFAULT_DEEPAGENT_INTERRUPT_ON
    assert captured["name"] == "deepagent"
    assert workflow.config == {"recursion_limit": 17}


def test_constructor_mounts_virtual_local_backend_at_workspace(
    monkeypatch,
    tmp_path,
):
    captured = {}
    workflow = _FakeWorkflow()
    backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=True, env={})

    monkeypatch.setattr(
        "chemgraph.graphs.deep_agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or workflow,
    )

    construct_deep_agent_graph(object(), backend=backend)

    mounted = captured["backend"]
    assert isinstance(mounted, CompositeBackend)
    assert mounted.default is backend
    assert mounted.routes == {"/workspace/": backend}

    write_result = mounted.write(
        "/workspace/probe.py",
        "print('workspace reachable')\n",
    )
    assert write_result.error is None
    assert (tmp_path / "probe.py").is_file()
    assert not (tmp_path / "workspace" / "probe.py").exists()

    command = (
        f"{shlex.quote(sys.executable)} "
        f"{shlex.quote(str(tmp_path / 'probe.py'))}"
    )
    execute_result = mounted.execute(command)
    assert execute_result.exit_code == 0
    assert execute_result.output == "workspace reachable\n"


def test_virtual_workspace_supplies_shell_host_path_to_model(tmp_path):
    model = _RecordingChatModel(
        responses=[AIMessage(content="workspace reviewed")]
    )
    workflow = construct_deep_agent_graph(
        model,
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True, env={}),
    )

    workflow.invoke(
        {"messages": [HumanMessage(content="Review the workspace")]},
        config={"configurable": {"thread_id": "workspace-mapping"}},
    )

    system_prompt = "\n".join(
        str(message.content)
        for message in model.received_messages
        if message.type == "system"
    )
    assert "Shell paths vs. virtual paths" in system_prompt
    assert "`/workspace/`" in system_prompt
    assert str(tmp_path.resolve()) in system_prompt


def test_constructor_preserves_nonvirtual_local_backend(monkeypatch, tmp_path):
    captured = {}
    workflow = _FakeWorkflow()
    backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=False, env={})

    monkeypatch.setattr(
        "chemgraph.graphs.deep_agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or workflow,
    )

    construct_deep_agent_graph(object(), backend=backend)

    assert captured["backend"] is backend


def test_constructor_supports_parent_checkpoint_and_unsafe_execution(monkeypatch):
    captured = {}
    workflow = _FakeWorkflow()
    backend = object()

    monkeypatch.setattr(
        "chemgraph.graphs.deep_agent.create_deep_agent",
        lambda **kwargs: captured.update(kwargs) or workflow,
    )

    construct_deep_agent_graph(
        object(),
        backend=backend,
        checkpointer=None,
        interrupt_on=None,
    )

    assert captured["backend"] is backend
    assert captured["checkpointer"] is None
    assert captured["interrupt_on"] is None


def test_constructor_rejects_nonpositive_recursion_limit():
    with pytest.raises(ValueError, match="must be positive"):
        construct_deep_agent_graph(object(), recursion_limit=0)


def test_standalone_graph_is_directly_invocable():
    workflow = construct_deep_agent_graph(
        _ScriptedChatModel(responses=[AIMessage(content="workspace reviewed")])
    )

    result = workflow.invoke(
        {"messages": [HumanMessage(content="Review the workspace")]},
        config={"configurable": {"thread_id": "direct-deep-agent"}},
    )

    assert result["messages"][-1].content == "workspace reviewed"


def test_chemgraph_routes_standalone_deep_agent_configuration(
    monkeypatch,
    tmp_path,
):
    captured = {}
    workflow = SimpleNamespace()
    backend = object()
    checkpointer = MemorySaver()
    tool = object()

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        lambda **_kwargs: (
            "fake-llm",
            PreparedModel(
                endpoint_name="test",
                protocol="openai_compatible",
                client_kwargs={},
            ),
        ),
    )

    def fake_constructor(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return workflow

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_deep_agent_graph",
        fake_constructor,
    )

    agent = ChemGraph(
        workflow_type="deep_agent",
        prompts=PromptConfig(deepagent="custom workspace prompt"),
        tools=[tool],
        deepagent_backend=backend,
        deepagent_auto_approve=True,
        checkpointer=checkpointer,
        enable_memory=False,
        log_dir=str(tmp_path),
    )

    assert agent.workflow is workflow
    assert captured["args"] == ("fake-llm",)
    assert captured["kwargs"] == {
        "tools": [tool],
        "system_prompt": "custom workspace prompt",
        "backend": backend,
        "recursion_limit": 50,
        "name": "deepagent",
        "checkpointer": checkpointer,
        "interrupt_on": None,
    }


def test_cli_resume_persists_deepagent_logs_and_session(
    monkeypatch,
    tmp_path,
):
    approval = {
        "action_requests": [
            {"name": "write_file", "args": {"file_path": "/workspace/result.txt"}}
        ],
        "review_configs": [
            {
                "action_name": "write_file",
                "allowed_decisions": ["approve", "reject"],
            }
        ],
    }

    class ApprovalWorkflow:
        def __init__(self):
            self.pending = False
            self.state = {"messages": []}

        async def astream(self, stream_input, **_kwargs):
            if isinstance(stream_input, Command):
                self.pending = False
                self.state = {
                    "messages": [
                        HumanMessage(content="Write the result"),
                        AIMessage(content="Result written."),
                    ]
                }
                yield self.state
                return

            self.pending = True
            self.state = {
                "messages": [HumanMessage(content="Write the result")]
            }
            yield {
                **self.state,
                "__interrupt__": [SimpleNamespace(value=approval)],
            }

        def get_state(self, _config):
            tasks = (
                (
                    SimpleNamespace(
                        interrupts=(SimpleNamespace(value=approval),)
                    ),
                )
                if self.pending
                else ()
            )
            return SimpleNamespace(values=self.state, tasks=tasks)

    workflow = ApprovalWorkflow()
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        lambda **_kwargs: (
            "fake-llm",
            PreparedModel(
                endpoint_name="test",
                protocol="openai_compatible",
                client_kwargs={},
            ),
        ),
    )
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_deep_agent_graph",
        lambda *_args, **_kwargs: workflow,
    )
    monkeypatch.setattr(
        commands,
        "_prompt_for_interrupt",
        lambda _payload: {"decisions": [{"type": "approve"}]},
    )

    log_dir = tmp_path / "cg_logs" / "session"
    workspace = tmp_path / "test"
    workspace.mkdir()
    agent = ChemGraph(
        workflow_type="deep_agent",
        deepagent_backend=LocalShellBackend(root_dir=workspace, env={}),
        memory_db_path=str(tmp_path / "sessions.db"),
        log_dir=str(log_dir),
    )

    result = commands.run_query(agent, "Write the result", thread_id=7)

    assert result.content == "Result written."
    state_logs = sorted(log_dir.glob("state_thread_7_*.json"))
    assert len(state_logs) == 2
    saved_states = [json.loads(path.read_text()) for path in state_logs]
    assert all(saved["thread_id"] == 7 for saved in saved_states)
    assert "Result written." not in state_logs[0].read_text()
    assert "Result written." in state_logs[1].read_text()
    assert not (workspace / "cg_logs").exists()

    session = agent.session_store.get_session(agent.session_id)
    assert session is not None
    assert [message.content for message in session.messages] == [
        "Write the result",
        "Result written.",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"workflow_type": "single_agent", "deepagent_backend": object()},
            "deepagent_backend",
        ),
        (
            {"workflow_type": "main_agent", "deepagent_auto_approve": True},
            "deepagent_auto_approve",
        ),
    ],
)
def test_chemgraph_rejects_deepagent_options_for_other_workflows(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ChemGraph(enable_memory=False, **kwargs)
