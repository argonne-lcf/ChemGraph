"""Inventory parity, attribution, offline CLI output, and catalog freshness."""

import asyncio
import importlib
import json
from pathlib import Path
import runpy
import sys

from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import LsResult
import pytest

from chemgraph.skills.catalog import list_skills, render_catalog
from chemgraph.skills.runtime import ChemGraphSkillsMiddleware, prepare_skill_backend
from tests.test_skill_lint import ROOT, skill


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    return tmp_path


@pytest.mark.parametrize("asynchronous", [False, True])
def test_catalog_matches_runtime_and_preserves_all_metadata(workspace, asynchronous):
    roots = [workspace / "home/.chemgraph/skills", workspace / ".agents/skills",
             workspace / "external one", workspace / "external two"]
    for index, root in enumerate(roots):
        root.mkdir(parents=True)
        skill(root, license="Apache-2.0", compatibility="Local host",
              metadata={"authors": f"Author {index}", "version": f"1.{index}",
                        "citation": "doi:example", "custom": "retained"},
              **{"allowed-tools": "read_file execute"})
    backend, sources, optional = prepare_skill_backend(
        CompositeBackend(default=StateBackend(), routes={
            "/workspace/": FilesystemBackend(root_dir=workspace, virtual_mode=True),
        }), (), skill_dirs=[str(path) for path in roots[2:]],
    )
    loader = ChemGraphSkillsMiddleware(backend=backend, sources=sources, optional=optional)
    state = (asyncio.run(loader.abefore_agent({}, None, {})) if asynchronous
             else loader.before_agent({}, None, {}))
    inventory = list_skills(workspace=workspace, skill_dirs=[str(path) for path in roots[2:]],
                            include_shadowed=True)
    active = [record for record in inventory["skills"] if record["active"]]
    assert [{key: record[key] for key in expected} for record, expected in
            zip(active, state["skills_metadata"], strict=True)] == state["skills_metadata"]
    candidates = [record for record in inventory["skills"] if record["name"] == "example"]
    assert [record["source_kind"] for record in candidates] == ["personal", "project", "explicit", "explicit"]
    assert [record["active"] for record in candidates] == [False, False, False, True]
    assert candidates[-1]["metadata"]["authors"] == "Author 3"
    assert candidates[-1]["host_path"] == str(roots[-1])
    assert candidates[-1]["allowed_tools"] == ["read_file", "execute"]
    for record in active[:2]:
        assert record["license"] == "Apache-2.0"
        author = {"chemgraph": "Thang Pham", "pbs-hpc": "Murat Keceli"}[record["name"]]
        assert record["metadata"] == {"authors": author, "maintainers": "tdpham2"}


def test_discovery_disabled_keeps_explicit_and_bundled(workspace):
    personal = workspace / "home/.chemgraph/skills"
    personal.mkdir(parents=True)
    skill(personal)
    external = workspace / "external"
    external.mkdir()
    result = list_skills(workspace=workspace, skill_dirs=[str(external)], discover=False)
    assert {s["name"] for s in result["skills"]} == {"chemgraph", "pbs-hpc"}
    skill(external)
    result = list_skills(workspace=workspace, skill_dirs=[str(external)], discover=False)
    assert {s["source_kind"] for s in result["skills"]} == {"bundled", "explicit"}


def test_project_override_replaces_bundled_attribution(workspace):
    root = workspace / ".agents/skills"
    root.mkdir(parents=True)
    skill(root, name="chemgraph", metadata={"authors": "Project author"}).rename(root / "chemgraph")
    effective = list_skills(workspace=workspace)["skills"]
    assert len(effective) == 2
    override = next(record for record in effective if record["name"] == "chemgraph")
    assert override["source_kind"] == "project" and override["active"]
    assert override["metadata"] == {"authors": "Project author"}
    assert override["license"] is None
    all_skills = list_skills(workspace=workspace, include_shadowed=True)["skills"]
    candidates = [record for record in all_skills if record["name"] == "chemgraph"]
    assert [record["active"] for record in candidates] == [False, True]
    assert candidates[0]["metadata"]["authors"] == "Thang Pham"


def test_missing_optional_and_invalid_individual_skills(workspace, caplog):
    assert len(list_skills(workspace=workspace)["skills"]) == 2
    assert not (workspace / ".agents").exists()
    root = workspace / ".agents/skills"
    root.mkdir(parents=True)
    skill(root, description=None).joinpath("SKILL.md").write_text("not YAML", encoding="utf-8")
    assert len(list_skills(workspace=workspace)["skills"]) == 2
    assert "failed metadata parse" in caplog.text


def test_optional_errors_warn_but_explicit_errors_fail(workspace, monkeypatch):
    root = workspace / ".agents/skills"
    root.mkdir(parents=True)
    original = FilesystemBackend.ls

    def blocked(self, path):
        if ".agents/skills" in path or self.cwd == root:
            return LsResult(error="blocked source")
        return original(self, path)

    monkeypatch.setattr(FilesystemBackend, "ls", blocked)
    result = list_skills(workspace=workspace)
    assert len(result["skills"]) == 2
    assert "blocked source" in result["warnings"][0]
    with pytest.raises(ValueError, match="blocked source"):
        list_skills(workspace=workspace, skill_dirs=[str(root)])


@pytest.mark.parametrize("missing", [False, True])
def test_list_cli_is_offline_and_returns_json(workspace, monkeypatch, capsys, missing):
    cli = importlib.import_module("chemgraph.cli.main")

    def no_agent(*args, **kwargs):
        pytest.fail("Listing must not start an agent or connect to MCP servers")

    monkeypatch.setattr(cli, "initialize_agent", no_agent)
    monkeypatch.setattr("chemgraph.models.loader.load_chat_model", no_agent)
    monkeypatch.setattr("chemgraph.cli.mcp_utils.load_mcp_tools_from_config", no_agent)
    argv = ["chemgraph", "skills", "list", "--workspace", str(workspace), "--json"]
    if missing:
        argv.extend(["--skill-dir", str(workspace / "absent")])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == int(missing)
    result = json.loads(capsys.readouterr().out)
    assert ("error" in result) == missing
    assert len(result["skills"]) == (0 if missing else 2)


def test_metadata_is_literal_in_terminal_and_markdown():
    from chemgraph.skills.cli import _text

    value = "[link=https://example.invalid]Name[/link]\x1b[31m"
    assert _text(value).plain == value.replace("\x1b", "\\x1b")
    assert not _text(value).spans
    table = render_catalog([{"name": "example", "description": "<script>x</script>|\nnext",
                             "metadata": {"authors": "[Name](url) *bold*"}}])
    assert "<script>" not in table and "[Name]" not in table and "*bold*" not in table
    assert "&#124;" in table and len(table.splitlines()) == 3


def test_bundled_catalog_freshness_and_regeneration(tmp_path):
    update = runpy.run_path(str(ROOT / "scripts/update_skill_catalog.py"))["update_catalog"]
    assert update(ROOT / "docs/skills.md", check=True)
    document = tmp_path / "skills.md"
    original = "before\n<!-- BEGIN BUNDLED SKILL CATALOG -->\nstale\n<!-- END BUNDLED SKILL CATALOG -->\nafter\n"
    document.write_text(original, encoding="utf-8")
    assert not update(document, check=True)
    assert document.read_text(encoding="utf-8") == original
    assert update(document)
    assert update(document, check=True)
    assert document.read_text(encoding="utf-8").endswith("after\n")
