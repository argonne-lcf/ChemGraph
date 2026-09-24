"""Tests for translating UI Deep Agent settings into ChemGraph options."""

import pytest

from chemgraph.agent.deepagent_backend import (
    DEEPAGENT_ENV_ALLOWLIST,
    create_host_shell_backend,
    resolve_workspace,
)
from ui.agent_manager import build_deepagent_options


def test_non_deep_agent_workflows_get_no_options(tmp_path):
    assert build_deepagent_options("single_agent", str(tmp_path), ["x"], False, ["run_ase"]) == {}


def test_deep_agent_backend_is_rooted_at_workspace(tmp_path, monkeypatch):
    from deepagents.backends import LocalShellBackend

    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("PATH", "/usr/bin")
    options = build_deepagent_options("deep_agent", str(tmp_path), None, True, None)

    backend = options["deepagent_backend"]
    assert isinstance(backend, LocalShellBackend)
    assert backend.virtual_mode is True
    assert str(backend.cwd) == str(tmp_path.resolve())
    assert options["deepagent_discover_skills"] is True
    assert "deepagent_skill_dirs" not in options
    assert "deepagent_tool_registry" not in options
    # Only the allowlisted variables reach shell commands.
    env = getattr(backend, "env", None) or backend._env
    assert "OPENAI_API_KEY" not in env
    assert env["PATH"] == "/usr/bin"
    assert set(env) <= set(DEEPAGENT_ENV_ALLOWLIST)


def test_deep_agent_skill_dirs_are_anchored_and_tools_restricted(tmp_path):
    skills = tmp_path / "skills"
    options = build_deepagent_options(
        "deep_agent", str(tmp_path), [str(skills), "rel/skills"], False, ["run_ase", "run_ase"]
    )
    anchored = options["deepagent_skill_dirs"]
    assert anchored[0] == str(skills.absolute())
    assert all(anchored_dir.startswith("/") or ":" in anchored_dir for anchored_dir in anchored)
    assert options["deepagent_discover_skills"] is False
    assert options["deepagent_tool_registry"].names() == ("run_ase",)


def test_deep_agent_empty_tool_list_disables_catalog(tmp_path):
    options = build_deepagent_options("deep_agent", str(tmp_path), None, True, [])
    assert options["deepagent_tool_registry"].names() == ()


def test_deep_agent_rejects_unknown_tool_and_missing_workspace(tmp_path):
    with pytest.raises(ValueError, match="Invalid Deep Agent tools"):
        build_deepagent_options("deep_agent", str(tmp_path), None, True, ["no_such_tool"])
    with pytest.raises(ValueError, match="not a directory"):
        build_deepagent_options("deep_agent", str(tmp_path / "missing"), None, True, None)


def test_shared_backend_helper_matches_cli_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace(None) == tmp_path.resolve()
    assert resolve_workspace("") == tmp_path.resolve()
    backend = create_host_shell_backend(None)
    assert str(backend.cwd) == str(tmp_path.resolve())
