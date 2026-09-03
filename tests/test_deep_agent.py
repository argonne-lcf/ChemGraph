"""Tests for the reusable standalone Deep Agent workflow."""

import hashlib
import json
import os
import shlex
import subprocess
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
    DEFAULT_DEEPAGENT_PROMPT,
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


def _legacy_topology_fingerprint(agent: ChemGraph) -> str:
    """Reproduce the main-agent fingerprint before skill support was added."""
    workspace = getattr(agent.deepagent_backend, "cwd", None)
    graph_config = agent.main_agent_metadata.graph_config
    topology_payload = {
        "model_name": agent.model_name,
        "reasoning_effort": agent.reasoning_effort,
        "recursion_limit": agent.recursion_limit,
        "structured_output": agent.structured_output,
        "generate_report": agent.generate_report,
        "max_retries": agent.max_retries,
        "human_supervised": agent.human_supervised,
        "terminal_tool_names": agent.terminal_tool_names,
        "enable_deepagent": agent.enable_deepagent,
        "workspace": str(workspace) if workspace is not None else None,
        "tool_signatures": graph_config.tool_signatures,
        "system_prompt": agent.system_prompt,
        "formatter_prompt": agent.formatter_prompt,
        "report_prompt": agent.report_prompt,
    }
    if (
        agent.enable_deepagent
        and agent.deepagent_prompt != DEFAULT_DEEPAGENT_PROMPT
    ):
        topology_payload["deepagent_prompt"] = agent.deepagent_prompt
    return hashlib.sha256(
        json.dumps(topology_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _message_content_text(message: Any) -> str:
    """Return text without escaping structured content-block strings."""
    content = message.content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", str(block))
            if isinstance(block, dict)
            else str(block)
            for block in content
        )
    return str(content)


def _shell_command(*parts: str) -> str:
    """Quote a command for the current platform's host shell."""
    if os.name == "nt":
        return subprocess.list2cmdline(parts)
    return shlex.join(parts)


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
    assert captured["skills"] is None
    assert isinstance(captured["backend"], StateBackend)
    assert isinstance(captured["checkpointer"], MemorySaver)
    assert captured["interrupt_on"] == DEFAULT_DEEPAGENT_INTERRUPT_ON
    assert captured["interrupt_on"] is not DEFAULT_DEEPAGENT_INTERRUPT_ON
    assert captured["name"] == "deepagent"
    assert workflow.config == {"recursion_limit": 17}


def test_constructor_loads_ordered_workspace_skills(tmp_path):
    base_skill = tmp_path / "base-skills" / "review-workflow"
    project_skill = tmp_path / "project-skills" / "review-workflow"
    base_skill.mkdir(parents=True)
    project_skill.mkdir(parents=True)
    base_skill.joinpath("SKILL.md").write_text(
        "---\n"
        "name: review-workflow\n"
        "description: Base review workflow.\n"
        "---\n\n"
        "# Base Review\n",
    )
    project_skill.joinpath("SKILL.md").write_text(
        "---\n"
        "name: review-workflow\n"
        "description: Project-specific review workflow.\n"
        "---\n\n"
        "# Project Review\n",
    )
    model = _RecordingChatModel(responses=[AIMessage(content="reviewed")])
    workflow = construct_deep_agent_graph(
        model,
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True, env={}),
        skills=[
            "/workspace/base-skills/",
            "/workspace/project-skills/",
        ],
    )

    workflow.invoke(
        {"messages": [HumanMessage(content="Review the project")]},
        config={"configurable": {"thread_id": "skill-loading"}},
    )

    system_prompt = "\n".join(
        _message_content_text(message)
        for message in model.received_messages
        if message.type == "system"
    )
    assert "Project-specific review workflow." in system_prompt
    assert "Base review workflow." not in system_prompt
    assert "/workspace/project-skills/review-workflow/SKILL.md" in system_prompt


@pytest.mark.parametrize(
    ("skills", "error"),
    [
        ("/skills/", TypeError),
        ([""], ValueError),
        ([object()], TypeError),
    ],
)
def test_constructor_rejects_invalid_skill_sources(skills, error):
    with pytest.raises(error, match="[Ss]kill"):
        construct_deep_agent_graph(object(), skills=skills)


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

    command = _shell_command(
        sys.executable,
        str(tmp_path / "probe.py"),
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
        _message_content_text(message)
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
        deepagent_skills=["/workspace/base/", "/workspace/project/"],
        deepagent_auto_approve=True,
        checkpointer=checkpointer,
        enable_memory=False,
        log_dir=str(tmp_path),
    )

    assert agent.workflow is workflow
    assert captured["args"] == ("fake-llm",)
    assert captured["kwargs"] == {
        "tools": [tool],
        "skills": ("/workspace/base/", "/workspace/project/"),
        "system_prompt": "custom workspace prompt",
        "backend": backend,
        "recursion_limit": 50,
        "name": "deepagent",
        "checkpointer": checkpointer,
        "interrupt_on": None,
    }


@pytest.mark.asyncio
async def test_chemgraph_resumes_all_pending_interrupt_ids(monkeypatch, tmp_path):
    first = SimpleNamespace(id="first-id", value={"question": "First?"})
    second = SimpleNamespace(id="second-id", value={"question": "Second?"})

    class MultiInterruptWorkflow:
        def __init__(self):
            self.pending = True
            self.resume_value = None
            self.state = {"messages": [HumanMessage(content="Continue")]}

        async def astream(self, stream_input, **_kwargs):
            if isinstance(stream_input, Command):
                self.resume_value = stream_input.resume
                self.pending = False
                self.state = {
                    "messages": [
                        HumanMessage(content="Continue"),
                        AIMessage(content="Completed."),
                    ]
                }
                yield self.state
                return
            yield {**self.state, "__interrupt__": [first, second]}

        def get_state(self, _config):
            tasks = (
                (SimpleNamespace(interrupts=(first, second)),)
                if self.pending
                else ()
            )
            return SimpleNamespace(
                values=self.state,
                tasks=tasks,
                interrupts=(),
            )

    workflow = MultiInterruptWorkflow()
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
    questions = []
    agent = ChemGraph(
        workflow_type="deep_agent",
        human_input_handler=lambda question: questions.append(question)
        or f"answer-{len(questions)}",
        enable_memory=False,
        log_dir=str(tmp_path),
    )

    result = await agent.run(
        "Continue",
        config={"configurable": {"thread_id": "multi-interrupt"}},
    )

    assert result.content == "Completed."
    assert questions == ["First?", "Second?"]
    assert workflow.resume_value == {
        "first-id": "answer-1",
        "second-id": "answer-2",
    }


@pytest.mark.asyncio
async def test_human_input_handler_receives_raw_structured_payload():
    payload = {
        "action_requests": [
            {"name": "execute", "args": {"command": "pytest"}}
        ],
        "review_configs": [
            {
                "action_name": "execute",
                "allowed_decisions": ["approve", "reject"],
            }
        ],
    }
    decision = {"decisions": [{"type": "reject"}]}
    received = []

    def handler(question, raw_payload):
        received.append((question, raw_payload))
        return decision

    agent = object.__new__(ChemGraph)
    agent.human_input_handler = handler

    result = await agent._call_human_input_handler(
        "Review the requested action.",
        payload=payload,
    )

    assert result is decision
    assert received == [("Review the requested action.", payload)]
    assert received[0][1] is payload


@pytest.mark.asyncio
async def test_human_input_handler_awaits_async_callable_object():
    payload = {"question": "Continue?"}

    class AsyncHandler:
        async def __call__(self, question, raw_payload):
            return question, raw_payload

    agent = object.__new__(ChemGraph)
    agent.human_input_handler = AsyncHandler()

    assert await agent._call_human_input_handler(
        "Continue?",
        payload=payload,
    ) == ("Continue?", payload)


@pytest.mark.asyncio
async def test_human_input_handler_does_not_retry_callback_type_error():
    calls = []

    def handler(question, payload):
        calls.append((question, payload))
        raise TypeError("handler failed")

    agent = object.__new__(ChemGraph)
    agent.human_input_handler = handler

    with pytest.raises(TypeError, match="handler failed"):
        await agent._call_human_input_handler("Continue?", payload={"id": 1})

    assert calls == [("Continue?", {"id": 1})]


def test_main_agent_metadata_persists_skills_in_topology(monkeypatch, tmp_path):
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
        "chemgraph.agent.llm_agent.construct_main_agent_graph",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    no_skill_agents = [
        ChemGraph(
            workflow_type="main_agent",
            enable_deepagent=enable_deepagent,
            enable_memory=False,
            log_dir=str(tmp_path),
        )
        for enable_deepagent in (False, True)
    ]
    for agent in no_skill_agents:
        graph_config = agent.main_agent_metadata.graph_config
        assert graph_config.deepagent_skills == ()
        assert graph_config.topology_fingerprint == _legacy_topology_fingerprint(
            agent
        )

    skill_agents = [
        ChemGraph(
            workflow_type="main_agent",
            enable_deepagent=True,
            deepagent_backend=SimpleNamespace(cwd=tmp_path),
            deepagent_skills=[skill],
            enable_memory=False,
            log_dir=str(tmp_path),
        )
        for skill in ("/workspace/base/", "/workspace/project/")
    ]

    configs = [agent.main_agent_metadata.graph_config for agent in skill_agents]
    assert configs[0].deepagent_skills == ("/workspace/base/",)
    assert configs[1].deepagent_skills == ("/workspace/project/",)
    assert configs[0].topology_fingerprint != configs[1].topology_fingerprint


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
        (
            {"workflow_type": "single_agent", "deepagent_skills": ["/skills/"]},
            "deepagent_skills",
        ),
    ],
)
def test_chemgraph_rejects_deepagent_options_for_other_workflows(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ChemGraph(enable_memory=False, **kwargs)
