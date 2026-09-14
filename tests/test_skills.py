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
from deepagents.backends.protocol import ExecuteResponse, LsResult, SandboxBackendProtocol
from deepagents.backends.utils import create_file_data
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
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
    ChemGraphSkillsMiddleware,
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


@pytest.mark.parametrize(
    "kind",
    ["shell", "filesystem", "nonvirtual", "composite", "composite-no-slash", "composite-both"],
)
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
    if kind.startswith("composite"):
        mounted = _normalize_backend(backend)
        routes = {"/workspace/" if kind == "composite" else "/workspace": backend}
        if kind == "composite-both":
            # The longer route must win, just as it does for file operations.
            routes["/workspace"] = FilesystemBackend(root_dir=home)
            routes["/workspace/"] = backend
        backend = CompositeBackend(default=mounted.default, routes=routes)
        original_routes = dict(routes)
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
    if kind.startswith("composite"):
        assert backend.routes == original_routes


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
    invoke(restored)
    assert "Updated skill" in _prompt(model) and "New skill" not in _prompt(model)
    assert restored.get_state(config).values["skills_load_errors"] == []


@pytest.mark.parametrize("asynchronous", [False, True])
def test_explicit_missing_source_errors_and_invalid_optional_warns(
    monkeypatch, tmp_path, caplog, asynchronous
):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    backend = LocalShellBackend(root_dir=tmp_path, env={})
    model = _RecordingChatModel(responses=[AIMessage(content="Done")] * 2)
    graph = construct_deep_agent_graph(
        model, backend=backend, skills=["/workspace/missing/"]
    )
    data = {"messages": [HumanMessage(content="List skills")]}

    def invoke(graph, thread):
        config = {"configurable": {"thread_id": thread}}
        return (
            asyncio.run(graph.ainvoke(data, config))
            if asynchronous else graph.invoke(data, config)
        )

    with pytest.raises(ValueError, match="Cannot load skills.*missing"):
        invoke(graph, "missing")
    assert not model.received_messages
    (tmp_path / "missing").mkdir()
    invoke(graph, "empty-directory")
    assert "pbs-hpc" in _prompt(model)
    _skill(tmp_path / ".agents/skills", "broken").write_text("no frontmatter")
    graph = construct_deep_agent_graph(model, backend=backend)
    invoke(graph, "optional")
    assert "failed metadata parse" in caplog.text
    assert "pbs-hpc" in _prompt(model)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_optional_symlink_failure_and_recovery(monkeypatch, tmp_path, caplog, asynchronous):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "home"))
    _skill(tmp_path / "home/.chemgraph/skills", "personal", "Personal instructions")
    workspace = tmp_path / "workspace"
    (workspace / ".agents").mkdir(parents=True)
    _skill(tmp_path / "shared", "shared", "Outside instructions")
    source = workspace / ".agents/skills"
    try:
        source.symlink_to(tmp_path / "shared", target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    backend = LocalShellBackend(root_dir=workspace, env={})
    model = _RecordingChatModel(responses=[AIMessage(content="Done")] * 2)
    saver = InMemorySaver()
    graph = construct_deep_agent_graph(model, backend=backend, checkpointer=saver)
    config = {"configurable": {"thread_id": "symlink"}}
    data = {"messages": [HumanMessage(content="Hello")]}

    def invoke(graph):
        return (
            asyncio.run(graph.ainvoke(data, config))
            if asynchronous else graph.invoke(data, config)
        )

    invoke(graph)
    assert "pbs-hpc" in _prompt(model) and "Personal instructions" in _prompt(model)
    assert "Outside instructions" not in _prompt(model)
    errors = graph.get_state(config).values["skills_load_errors"]
    assert len(errors) == 1 and "/workspace/.agents/skills/" in errors[0]
    assert "outside root directory" in caplog.text

    required_model = _RecordingChatModel(responses=[AIMessage(content="Done")])
    required = construct_deep_agent_graph(
        required_model, backend=backend, skills=["/workspace/.agents/skills/"],
    )
    with pytest.raises(ValueError, match="Cannot load skills.*outside root directory"):
        invoke(required)
    assert not required_model.received_messages

    source.unlink()
    _skill(source, "repaired", "Repaired instructions")
    restored = construct_deep_agent_graph(model, backend=backend, checkpointer=saver)
    invoke(restored)
    assert "Repaired instructions" in _prompt(model)
    assert restored.get_state(config).values["skills_load_errors"] == []
    assert "outside root directory" not in _prompt(model)


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
        else StoreBackend(store=InMemoryStore(), namespace=lambda _: ("skills-test",))
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
    else:
        assert backend.write("/result.txt", "Stored result").error is None
        assert backend.read("/result.txt").file_data["content"] == "Stored result"
        assert default.read("/result.txt").file_data["content"] == "Stored result"
        assert backend.write(BUNDLED_SKILLS_PATH + "chemgraph/SKILL.md", "changed").error


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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("kind", ["state", "store"])
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("valid_skill", [False, True])
def test_required_virtual_sources_need_files(
    asynchronous, kind, routed, valid_skill, caplog,
):
    leaf = (
        StateBackend() if kind == "state" else
        StoreBackend(store=InMemoryStore(), namespace=lambda _: ("skills-test",))
    )
    backend = leaf
    source = "/extra/"
    if routed:
        backend = CompositeBackend(
            default=StateBackend(),
            routes={"/library/": CompositeBackend(
                default=StateBackend(), routes={"/nested/": leaf},
            )},
        )
        source = "/library/nested/extra/"
    model = _RecordingChatModel(responses=[
        _read_skill(source + "example/SKILL.md"), AIMessage(content="Done"),
    ])
    graph = construct_deep_agent_graph(model, backend=backend, skills=[source])
    config = {"configurable": {"thread_id": "required-source"}}

    def invoke(files=None):
        data = {"messages": [HumanMessage(content="Read the configured skill")]}
        if files is not None:
            data["files"] = files
        return (
            asyncio.run(graph.ainvoke(data, config))
            if asynchronous else graph.invoke(data, config)
        )

    with pytest.raises(ValueError, match="Cannot load skills.*missing or empty") as exc:
        invoke()
    assert source in str(exc.value)
    assert not model.received_messages

    path = "/extra/example/SKILL.md"
    content = "Seeded instructions"
    if valid_skill:
        content = "---\nname: example\ndescription: Seeded instructions\n---\n" + content
    files = {path: create_file_data(content)} if kind == "state" else None
    if kind == "store":
        assert leaf.write(path, content).error is None
    state = invoke(files)
    assert any(m.type == "tool" and "Seeded instructions" in str(m.content)
               for m in state["messages"])
    values = graph.get_state(config).values
    assert ("example" in {s["name"] for s in values["skills_metadata"]}) == valid_skill
    assert values["skills_load_errors"] == []
    if not valid_skill:
        assert "failed metadata parse" in caplog.text
    assert "pbs-hpc" in _prompt(model)

    if kind == "store":
        assert leaf.delete(path).error is None
    calls = model.response_index
    with pytest.raises(ValueError, match="Cannot load skills.*missing or empty"):
        invoke({path: None} if kind == "state" else None)
    assert model.response_index == calls


def test_empty_custom_source_keeps_backend_semantics():
    class EmptyBackend(BundledSkillsBackend):
        def ls(self, path):
            return LsResult(entries=[])

    model = _RecordingChatModel(responses=[AIMessage(content="Done")])
    graph = construct_deep_agent_graph(model, backend=EmptyBackend(), skills=["/extra/"])
    graph.invoke(
        {"messages": [HumanMessage(content="List skills")]},
        {"configurable": {"thread_id": "custom-empty"}},
    )
    assert "pbs-hpc" in _prompt(model)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "error_type", [OSError, ValueError, RuntimeError, TypeError, GraphInterrupt, asyncio.CancelledError],
)
def test_optional_download_failure_handling(tmp_path, asynchronous, error_type):
    class FailingBackend(FilesystemBackend):
        def download_files(self, paths):
            raise error_type("Download failed")

    _skill(tmp_path / ".agents/skills")
    backend, sources, optional = prepare_skill_backend(
        FailingBackend(root_dir=tmp_path), (), user_skills_dir=str(tmp_path / "missing"),
    )
    middleware = ChemGraphSkillsMiddleware(backend=backend, sources=sources, optional=optional)

    def load():
        return (
            asyncio.run(middleware.abefore_agent({}, None, {}))
            if asynchronous else middleware.before_agent({}, None, {})
        )

    if error_type in (OSError, ValueError, RuntimeError):
        update = load()
        assert {s["name"] for s in update["skills_metadata"]} == {"chemgraph", "pbs-hpc"}
        assert len(update["skills_load_errors"]) == 1
        assert "Cannot load skills from '/.agents/skills/': Download failed" in update["skills_load_errors"]
    else:
        with pytest.raises(error_type):
            load()


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
