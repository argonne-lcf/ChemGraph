"""Host skill collections outside the Deep Agent workspace."""

import asyncio

import pytest
from deepagents.backends import CompositeBackend, LocalShellBackend, StateBackend
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from chemgraph.graphs.deep_agent import _normalize_backend, construct_deep_agent_graph
from chemgraph.skills.runtime import (
    EXTERNAL_SKILLS_PATH,
    prepare_skill_backend,
    resolve_skill_dirs,
)
from tests.test_deep_agent import _RecordingChatModel
from tests.test_skills import _prompt, _read_skill, _skill


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
    assert str(root) in str(exc.value)


class _SkillEvidenceModel(_RecordingChatModel):
    tool_results: list[str] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.tool_results.extend(str(m.content) for m in messages if m.type == "tool")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("delegated", [False, True])
def test_external_atomistic_skill_and_helpers_are_readable(
    monkeypatch, tmp_path, asynchronous, delegated
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    collection = tmp_path / "external/AtomisticSkills/.agents/skills"
    widom = _skill(collection, "chem-sorption-widom", "Widom insertion fixture")
    widom.write_text(
        widom.read_text() + "Environment: mace-agent; helper: scripts/run_widom.py\n"
    )
    _skill(collection, "chem-sorption-relax", "Sibling relaxation fixture")
    scripts = widom.parent / "scripts"
    scripts.mkdir()
    (scripts / "run_widom.py").write_text("# Widom helper fixture, never executed\n")
    monkeypatch.chdir(workspace)
    roots = resolve_skill_dirs(["../external/AtomisticSkills/.agents/skills/"])
    shell = LocalShellBackend(root_dir=workspace, env={})
    backend, sources, _ = prepare_skill_backend(
        _normalize_backend(shell),
        (),
        discover_skills=False,
        skill_dirs=roots,
    )
    source = sources[-1]
    assert source.startswith(EXTERNAL_SKILLS_PATH)
    responses = [
        _read_skill(source + "chem-sorption-widom/SKILL.md", "widom"),
        _read_skill(source + "chem-sorption-widom/scripts/run_widom.py", "helper"),
        _read_skill(source + "chem-sorption-relax/SKILL.md", "sibling"),
        AIMessage(content="Read the external skill and its resources"),
    ]
    if delegated:
        responses.insert(
            0,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "subagent_type": "general-purpose",
                            "description": "Read the Widom skill and resources",
                        },
                        "id": "delegate",
                        "type": "tool_call",
                    }
                ],
            ),
        )
        responses.append(AIMessage(content="Done"))
    model = _SkillEvidenceModel(responses=responses)
    graph = construct_deep_agent_graph(
        model,
        backend=shell,
        skill_dirs=roots,
        discover_skills=False,
    )
    data = {
        "messages": [
            HumanMessage(content="Read the Widom skill before running anything")
        ]
    }
    config = {"configurable": {"thread_id": "external"}}
    if asynchronous:
        asyncio.run(graph.ainvoke(data, config))
    else:
        graph.invoke(data, config)
    state = graph.get_state(config).values
    assert "chem-sorption-widom" in {s["name"] for s in state["skills_metadata"]}
    assert state["skills_load_errors"] == []
    evidence = "\n".join(model.tool_results)
    assert "Environment: mace-agent" in evidence
    assert "Widom helper fixture" in evidence
    assert "Sibling relaxation fixture" in evidence
    assert f"`{source}` -> `{collection}/`" in _prompt(model)
    assert not list(workspace.iterdir())
    assert backend.default.shell is shell
    # Stable mounts survive changes to cwd and retain the same original files.
    monkeypatch.chdir(tmp_path)
    restored, restored_sources, _ = prepare_skill_backend(
        StateBackend(), (), skill_dirs=roots
    )
    assert restored_sources[-1] == source
    assert restored.read(source + "chem-sorption-widom/SKILL.md").error is None


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
