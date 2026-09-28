"""Host skill collections outside the Deep Agent workspace."""

import pytest
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from langchain_core.messages import AIMessage, HumanMessage

from chemgraph.graphs.deep_agent import _normalize_backend, construct_deep_agent_graph
from chemgraph.skills.runtime import (
    EXTERNAL_SKILLS_PATH,
    prepare_skill_backend,
    resolve_skill_dirs,
)
from tests.test_deep_agent import _RecordingChatModel
from tests.test_skills import _prompt, _skill


@pytest.mark.parametrize(
    "form", ["absolute", "relative", "dot", "parent", "home", "link"]
)
def test_host_paths_are_canonical_and_last_duplicate_wins(monkeypatch, tmp_path, form):
    root = tmp_path / "skills with spaces"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "link").symlink_to(root, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    value = {
        "absolute": str(root),
        "relative": root.name,
        "dot": f"./{root.name}/",
        "parent": f"../{root.name}",
        "home": f"~/{root.name}",
        "link": "link",
    }[form]
    if form == "parent":
        monkeypatch.chdir(work)
    assert resolve_skill_dirs([str(root), str(other), value]) == (str(other), str(root))


@pytest.mark.parametrize("kind", ["missing", "file", "unreadable"])
def test_invalid_host_directory_identifies_input_and_resolved_path(
    monkeypatch, tmp_path, kind
):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / kind
    if kind == "file":
        root.write_text("not a directory")
    elif kind == "unreadable":
        root.mkdir()

        def denied(_path):
            raise PermissionError("denied")

        monkeypatch.setattr("chemgraph.skills.runtime.os.scandir", denied)
    with pytest.raises(ValueError) as exc:
        resolve_skill_dirs([f"./{kind}"])
    assert f"./{kind}" in str(exc.value)
    assert repr(str(root)) in str(exc.value)


def test_external_skill_resources_keep_stable_mounts(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    collection = tmp_path / "external/skills"
    skill = _skill(collection, "example")
    helper = skill.parent / "helper.py"
    helper.write_text("# helper resource\n")
    shell = LocalShellBackend(root_dir=workspace, env={})
    backend, sources, _ = prepare_skill_backend(
        _normalize_backend(shell), (), discover_skills=False,
        skill_dirs=(str(collection),),
    )
    source = sources[-1]
    assert source.startswith(EXTERNAL_SKILLS_PATH)
    assert backend.default.shell is shell
    for resource in (skill, helper):
        mounted = source + resource.relative_to(collection).as_posix()
        assert backend.read(mounted).file_data["content"] == resource.read_text()
    monkeypatch.chdir(workspace)
    restored, restored_sources, _ = prepare_skill_backend(
        StateBackend(), (), skill_dirs=(str(collection),)
    )
    assert restored_sources[-1] == source
    assert restored.read(source + "example/helper.py").file_data["content"] == helper.read_text()
    assert not list(workspace.iterdir())


@pytest.mark.parametrize("with_backend_source", [False, True])
def test_host_sources_override_discovery_and_preserve_backend_precedence(
    tmp_path, with_backend_source
):
    _skill(tmp_path / ".agents/skills", description="Project instructions")
    _skill(tmp_path / "first", description="First host instructions")
    _skill(tmp_path / "last", description="Last host instructions")
    _skill(tmp_path / "virtual", description="Backend instructions")
    model = _RecordingChatModel(responses=[AIMessage(content="Done")])
    graph = construct_deep_agent_graph(
        model,
        backend=LocalShellBackend(root_dir=tmp_path, env={}),
        user_skills_dir=str(tmp_path / "personal"),
        skill_dirs=[str(tmp_path / "first"), str(tmp_path / "last")],
        skills=["/workspace/virtual/"] if with_backend_source else None,
    )
    graph.invoke(
        {"messages": [HumanMessage(content="List skills")]},
        {"configurable": {"thread_id": "precedence"}},
    )
    expected = (
        "Backend instructions" if with_backend_source else "Last host instructions"
    )
    assert expected in _prompt(model)
    assert "First host instructions" not in _prompt(model)
    assert "Project instructions" not in _prompt(model)


def test_external_routes_preserve_caller_backend_and_reject_conflicts(tmp_path):
    default = StateBackend()
    caller = CompositeBackend(
        default=default,
        routes={"/existing/": StateBackend()},
        artifacts_root="/existing/",
    )
    backend, _, _ = prepare_skill_backend(caller, (), skill_dirs=[str(tmp_path)])
    assert backend.default is default
    assert backend.artifacts_root == "/existing/"
    assert backend.routes["/existing/"] is caller.routes["/existing/"]
    assert list(caller.routes) == ["/existing/"]
    conflicting = CompositeBackend(
        default=default, routes={EXTERNAL_SKILLS_PATH: default}
    )
    with pytest.raises(ValueError, match="conflicts with reserved"):
        prepare_skill_backend(conflicting, (), skill_dirs=[str(tmp_path)])
