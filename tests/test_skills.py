"""Portable skill resources, source precedence, and checkpoint discovery."""

import asyncio
from pathlib import Path

import pytest
from deepagents.backends import (
    CompositeBackend,
    FilesystemBackend,
    LocalShellBackend,
    StateBackend,
    StoreBackend,
)
from deepagents.backends.protocol import ExecuteResponse, SandboxBackendProtocol
from deepagents.backends.utils import create_file_data
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from chemgraph.graphs.deep_agent import (
    DEFAULT_DEEPAGENT_PROMPT,
    DEFAULT_DEEPAGENT_WORKSPACE_PROMPT,
    _normalize_backend,
    construct_deep_agent_graph,
)
from chemgraph.skills.backend import BundledSkillsBackend
from chemgraph.skills.runtime import (
    BUNDLED_SKILLS_PATH,
    USER_SKILLS_PATH,
    prepare_skill_backend,
)
from tests.test_deep_agent import _RecordingChatModel, _message_content_text


def _skill(root, name="chemgraph", description="Project chemistry instructions"):
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(f"---\nname: {name}\ndescription: {description}\n---\n\n# {name}\n")
    return path


def _prompt(model):
    return "\n".join(
        _message_content_text(m) for m in model.received_messages if m.type == "system"
    )


def _read_skill(path="/chemgraph-skills/chemgraph/SKILL.md", call_id="read-skill"):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "read_file",
                "args": {"file_path": path, "limit": 1000},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_default_state_backend_reads_bundled_skill_without_seeding(asynchronous):
    model = _RecordingChatModel(
        responses=[_read_skill(), AIMessage(content="Read the skill")]
    )
    graph = construct_deep_agent_graph(model)
    data = {"messages": [HumanMessage(content="Read the ChemGraph skill")]}
    config = {"configurable": {"thread_id": "builtin"}}
    state = (
        asyncio.run(graph.ainvoke(data, config))
        if asynchronous
        else graph.invoke(data, config)
    )
    assert "pbs-hpc" in _prompt(model)
    assert any(
        m.type == "tool" and "Use ChemGraph" in str(m.content)
        for m in state["messages"]
    )
    assert not any(
        path.startswith(BUNDLED_SKILLS_PATH) for path in state.get("files", {})
    )


@pytest.mark.parametrize("kind", ["shell", "filesystem", "nonvirtual", "composite"])
def test_local_source_precedence_and_discovery(monkeypatch, tmp_path, kind):
    home = tmp_path / "home"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    _skill(home / ".chemgraph/skills", description="Personal chemistry instructions")
    project = _skill(workspace / ".agents/skills")
    cls = FilesystemBackend if kind == "filesystem" else LocalShellBackend
    kwargs = {} if cls is FilesystemBackend else {"env": {}}
    backend = cls(root_dir=workspace, virtual_mode=kind != "nonvirtual", **kwargs)
    if kind == "composite":
        backend = _normalize_backend(backend)
    model = _RecordingChatModel(responses=[AIMessage(content="Done")] * 3)
    graph = construct_deep_agent_graph(model, backend=backend)
    config = {"configurable": {"thread_id": kind}}
    data = {"messages": [HumanMessage(content="Inspect skills")]}
    graph.invoke(data, config)
    assert "Project chemistry instructions" in _prompt(model)
    assert "Personal chemistry instructions" not in _prompt(model)
    project.unlink()
    graph.invoke(data, config)
    assert "Personal chemistry instructions" in _prompt(model)
    explicit = _skill(workspace / "site", description="Explicit chemistry instructions")
    prefix = (
        str(workspace)
        if kind == "nonvirtual"
        else ""
        if kind == "filesystem"
        else "/workspace"
    )
    override = construct_deep_agent_graph(
        model, backend=backend, skills=[prefix + "/site/"]
    )
    override.invoke(data, {"configurable": {"thread_id": "override"}})
    assert "Explicit chemistry instructions" in _prompt(model)
    assert str(explicit.name) in _prompt(model)


def test_discovery_disabled_keeps_bundled_and_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    _skill(tmp_path / ".chemgraph/skills", "personal", "Personal skill")
    _skill(tmp_path / ".agents/skills", "project", "Project skill")
    _skill(tmp_path / "extra", "explicit", "Explicit skill")
    model = _RecordingChatModel(responses=[AIMessage(content="Done")])
    graph = construct_deep_agent_graph(
        model,
        backend=LocalShellBackend(root_dir=tmp_path, env={}),
        discover_skills=False,
        skills=["/workspace/extra/"],
    )
    graph.invoke(
        {"messages": [HumanMessage(content="List skills")]},
        {"configurable": {"thread_id": "explicit"}},
    )
    prompt = _prompt(model)
    assert "pbs-hpc" in prompt and "Explicit skill" in prompt
    assert "Personal skill" not in prompt and "Project skill" not in prompt


@pytest.mark.parametrize("asynchronous", [False, True])
def test_new_optional_directory_and_restored_catalog(
    monkeypatch, tmp_path, asynchronous
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    saver = InMemorySaver()
    backend = LocalShellBackend(root_dir=tmp_path, env={})
    model = _RecordingChatModel(responses=[AIMessage(content="Done")] * 3)
    graph = construct_deep_agent_graph(model, backend=backend, checkpointer=saver)
    config = {"configurable": {"thread_id": "refresh"}}
    data = {"messages": [HumanMessage(content="Inspect skills")]}

    def invoke(graph):
        return (
            asyncio.run(graph.ainvoke(data, config))
            if asynchronous
            else graph.invoke(data, config)
        )

    invoke(graph)
    assert "New skill" not in _prompt(model)
    _skill(tmp_path / ".agents/skills", "new-skill", "New skill")
    invoke(graph)
    assert "New skill" in _prompt(model)
    _skill(tmp_path / ".agents/skills", "new-skill", "Updated skill")
    restored = construct_deep_agent_graph(model, backend=backend, checkpointer=saver)
    state = invoke(restored)
    assert "Updated skill" in _prompt(model) and "New skill" not in _prompt(model)
    assert not state.get("skills_load_errors")


def test_explicit_missing_source_errors_and_invalid_optional_warns(
    monkeypatch, tmp_path, caplog
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    backend = LocalShellBackend(root_dir=tmp_path, env={})
    model = _RecordingChatModel(responses=[AIMessage(content="Done")])
    graph = construct_deep_agent_graph(
        model, backend=backend, skills=["/workspace/missing/"]
    )
    data = {"messages": [HumanMessage(content="List skills")]}
    with pytest.raises(ValueError, match="Cannot load skills.*missing"):
        graph.invoke(data, {"configurable": {"thread_id": "missing"}})
    _skill(tmp_path / ".agents/skills", "broken").write_text("no frontmatter")
    graph = construct_deep_agent_graph(model, backend=backend)
    graph.invoke(data, {"configurable": {"thread_id": "optional"}})
    assert "failed metadata parse" in caplog.text
    assert "pbs-hpc" in _prompt(model)


@pytest.mark.parametrize(
    "route", [BUNDLED_SKILLS_PATH, USER_SKILLS_PATH, "/chemgraph-skills/child/", "/"]
)
def test_reserved_route_collision(route):
    backend = CompositeBackend(default=StateBackend(), routes={route: StateBackend()})
    with pytest.raises(ValueError, match="reserved"):
        prepare_skill_backend(backend, ())


class _RemoteBackend(BundledSkillsBackend, SandboxBackendProtocol):
    """A fake remote executor whose filesystem is unrelated to the host."""

    @property
    def id(self):
        return "remote-test"

    def execute(self, command, *, timeout=None):
        return ExecuteResponse(output=f"remote:{command}", exit_code=0)


@pytest.mark.parametrize("kind", ["remote", "store"])
def test_nonlocal_backends_keep_execution_and_routes_without_host_discovery(
    monkeypatch, kind
):
    def forbidden_home(cls):
        raise AssertionError("A nonlocal backend must not inspect the user's home")

    monkeypatch.setattr(Path, "home", classmethod(forbidden_home))
    default = (
        _RemoteBackend()
        if kind == "remote"
        else StoreBackend(store=InMemoryStore(), namespace=("skills-test",))
    )
    caller = CompositeBackend(
        default=default, routes={"/notes/": StateBackend()}, artifacts_root="/notes/"
    )
    backend, sources, optional = prepare_skill_backend(caller, ())
    assert backend.default is default
    assert backend.routes["/notes/"] is caller.routes["/notes/"]
    assert backend.artifacts_root == "/notes/"
    assert sources == [BUNDLED_SKILLS_PATH] and optional == {}
    assert BUNDLED_SKILLS_PATH not in caller.routes
    assert backend.read(BUNDLED_SKILLS_PATH + "chemgraph/SKILL.md").error is None
    if kind == "remote":
        assert backend.execute("pwd").output == "remote:pwd"


def test_skill_reads_work_inside_default_child_agent():
    task = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "task",
                "args": {
                    "subagent_type": "general-purpose",
                    "description": "Read the ChemGraph skill",
                },
                "id": "child",
                "type": "tool_call",
            }
        ],
    )
    model = _RecordingChatModel(
        responses=[
            task,
            _read_skill(),
            AIMessage(content="Read ChemGraph instructions"),
            AIMessage(content="Done"),
        ]
    )
    graph = construct_deep_agent_graph(model)
    state = graph.invoke(
        {"messages": [HumanMessage(content="Delegate a skill read")]},
        {"configurable": {"thread_id": "child"}},
    )
    assert any(
        m.type == "tool" and "Read ChemGraph instructions" in str(m.content)
        for m in state["messages"]
    )


def test_refresh_does_not_replay_pending_approved_write(tmp_path):
    write = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "write_file",
                "args": {"file_path": "/workspace/result.txt", "content": "once"},
                "id": "write",
                "type": "tool_call",
            }
        ],
    )
    model = _RecordingChatModel(
        responses=[_read_skill(), write, AIMessage(content="Done")]
    )
    saver = InMemorySaver()
    backend = LocalShellBackend(root_dir=tmp_path, env={})
    kwargs = {"backend": backend, "checkpointer": saver, "discover_skills": False}
    graph = construct_deep_agent_graph(model, **kwargs)
    config = {"configurable": {"thread_id": "approval"}}
    state = graph.invoke({"messages": [HumanMessage(content="Write a result")]}, config)
    assert state["__interrupt__"] and not (tmp_path / "result.txt").exists()
    restored = construct_deep_agent_graph(model, **kwargs)
    state = restored.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}), config
    )
    assert (tmp_path / "result.txt").read_text() == "once"
    assert (
        len(
            [
                m
                for m in state["messages"]
                if m.type == "tool" and m.name == "write_file"
            ]
        )
        == 1
    )


def test_default_roles_and_custom_prompt_are_distinct(monkeypatch):
    from chemgraph.agent.llm_agent import ChemGraph, PromptConfig
    from chemgraph.models.endpoints import PreparedModel

    model = _RecordingChatModel(responses=[AIMessage(content="Done")])
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        lambda **_: (
            model,
            PreparedModel(
                endpoint_name="test", protocol="openai_compatible", client_kwargs={}
            ),
        ),
    )
    for workflow, expected in [
        ("deep_agent", DEFAULT_DEEPAGENT_PROMPT),
        ("main_agent", DEFAULT_DEEPAGENT_WORKSPACE_PROMPT),
    ]:
        agent = ChemGraph(workflow_type=workflow, enable_memory=False)
        assert agent.deepagent_prompt == expected
        agent = ChemGraph(
            workflow_type=workflow,
            prompts=PromptConfig(deepagent="Custom instructions"),
            enable_memory=False,
        )
        assert agent.deepagent_prompt == "Custom instructions"


def test_state_skill_sources_remain_supported():
    model = _RecordingChatModel(responses=[AIMessage(content="Done")])
    graph = construct_deep_agent_graph(model, skills=["/extra/"])
    graph.invoke(
        {
            "messages": [HumanMessage(content="Inspect skills")],
            "files": {
                "/extra/state-skill/SKILL.md": create_file_data(
                    "---\nname: state-skill\ndescription: Seeded instructions\n---\nBody\n"
                )
            },
        },
        {"configurable": {"thread_id": "state-extra"}},
    )
    assert "Seeded instructions" in _prompt(model)


def test_personal_root_and_discovery_are_persisted(monkeypatch, tmp_path):
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.memory.schemas import MainAgentGraphConfig
    from chemgraph.models.endpoints import PreparedModel

    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        lambda **_: (
            object(),
            PreparedModel(
                endpoint_name="test", protocol="openai_compatible", client_kwargs={}
            ),
        ),
    )
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.construct_main_agent_graph",
        lambda *_, **__: object(),
    )
    original_home = tmp_path / "original-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: original_home))
    backend = LocalShellBackend(root_dir=tmp_path, env={})
    kwargs = dict(
        workflow_type="main_agent",
        enable_deepagent=True,
        deepagent_backend=backend,
        enable_memory=False,
        log_dir=str(tmp_path),
    )
    agent = ChemGraph(**kwargs)
    saved = agent.main_agent_metadata.graph_config
    assert saved.deepagent_discover_skills is True
    assert saved.deepagent_user_skills_dir == str(original_home / ".chemgraph/skills")
    monkeypatch.setattr(
        Path, "home", classmethod(lambda cls: tmp_path / "changed-home")
    )
    restored = ChemGraph(
        **kwargs,
        deepagent_discover_skills=saved.deepagent_discover_skills,
        deepagent_user_skills_dir=saved.deepagent_user_skills_dir,
    )
    assert (
        restored.main_agent_metadata.graph_config.topology_fingerprint
        == saved.topology_fingerprint
    )
    changed = ChemGraph(**kwargs, deepagent_discover_skills=False)
    assert (
        changed.main_agent_metadata.graph_config.topology_fingerprint
        != saved.topology_fingerprint
    )
    legacy = MainAgentGraphConfig(model_name="test")
    assert legacy.deepagent_discover_skills is False
    assert legacy.deepagent_user_skills_dir is None


def test_standalone_reads_skill_then_uses_attached_chemistry_tool():
    from langchain_core.tools import tool

    calls = []

    @tool
    def run_ase_single(input_structure_file: str) -> dict:
        """Return a hermetic calculation result for a scripted workflow."""
        calls.append(input_structure_file)
        return {"potential_energy": 1.25, "energy_unit": "eV", "status": "completed"}

    calculation = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "run_ase_single",
                "args": {"input_structure_file": "/server/water.xyz"},
                "id": "calculation",
                "type": "tool_call",
            }
        ],
    )
    model = _RecordingChatModel(
        responses=[_read_skill(), calculation, AIMessage(content="Energy: 1.25 eV")]
    )
    graph = construct_deep_agent_graph(model, tools=[run_ase_single])
    state = graph.invoke(
        {"messages": [HumanMessage(content="Calculate the energy")]},
        {"configurable": {"thread_id": "chemistry"}},
    )
    assert calls == ["/server/water.xyz"]
    assert [m.name for m in state["messages"] if m.type == "tool"] == [
        "read_file",
        "run_ase_single",
    ]
    assert '"energy_unit": "eV"' in str(state["messages"][-2].content)
    assert "standalone Deep Agent" in _prompt(model)


def test_optional_permission_error_does_not_hide_bundled_catalog(caplog):
    from chemgraph.skills.runtime import ChemGraphSkillsMiddleware

    class InaccessibleDirectory:
        def stat(self):
            raise PermissionError("permission denied")

    backend, sources, optional = prepare_skill_backend(StateBackend(), ())
    middleware = ChemGraphSkillsMiddleware(
        backend=backend,
        sources=[*sources, "/optional/"],
        optional={"/optional/": InaccessibleDirectory()},
    )
    update = middleware.before_agent({}, None, {})
    assert {skill["name"] for skill in update["skills_metadata"]} == {
        "chemgraph",
        "pbs-hpc",
    }
    assert "Cannot inspect optional skills" in caplog.text
