"""Tests for translating UI Deep Agent settings into ChemGraph options."""

import pytest

from chemgraph.agent.deepagent_backend import (
    DEEPAGENT_ENV_ALLOWLIST,
    create_host_shell_backend,
    resolve_workspace,
)
from ui import deepagent_policy
from ui.agent_manager import build_deepagent_options


@pytest.fixture(autouse=True)
def _deep_agent_enabled_in(monkeypatch, tmp_path):
    """Operator enabled the Deep Agent with tmp_path as the only root."""
    monkeypatch.setenv(deepagent_policy.ENABLE_ENV, "1")
    monkeypatch.setenv(deepagent_policy.ROOTS_ENV, str(tmp_path))


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


def test_deep_agent_skill_dirs_are_confined_and_tools_restricted(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    (tmp_path / "rel" / "skills").mkdir(parents=True)
    options = build_deepagent_options(
        "deep_agent", str(tmp_path), [str(skills), "rel/skills"], False, ["run_ase", "run_ase"]
    )
    anchored = options["deepagent_skill_dirs"]
    # Relative entries are anchored at the first allowed root.
    assert anchored == (
        str(skills.resolve()), str((tmp_path / "rel" / "skills").resolve())
    )
    assert options["deepagent_discover_skills"] is False
    assert options["deepagent_tool_registry"].names() == ("run_ase",)


def test_deep_agent_empty_tool_list_disables_catalog(tmp_path):
    options = build_deepagent_options("deep_agent", str(tmp_path), None, True, [])
    assert options["deepagent_tool_registry"].names() == ()


def test_deep_agent_rejects_unknown_tool_and_missing_workspace(tmp_path):
    with pytest.raises(ValueError, match="Invalid Deep Agent tools"):
        build_deepagent_options("deep_agent", str(tmp_path), None, True, ["no_such_tool"])
    with pytest.raises(ValueError, match="does not exist"):
        build_deepagent_options("deep_agent", str(tmp_path / "missing"), None, True, None)


def test_deep_agent_requires_operator_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv(deepagent_policy.ENABLE_ENV)
    with pytest.raises(ValueError, match="disabled on this server"):
        build_deepagent_options("deep_agent", str(tmp_path), None, True, None)


@pytest.mark.parametrize("escape", ["..", "/", "~", "ws/../../"])
def test_deep_agent_paths_must_stay_inside_allowed_roots(tmp_path, escape):
    (tmp_path / "ws").mkdir()
    with pytest.raises(ValueError, match="outside the directories"):
        build_deepagent_options("deep_agent", escape, None, True, None)
    with pytest.raises(ValueError, match="outside the directories"):
        build_deepagent_options("deep_agent", str(tmp_path), [escape], True, None)


def test_symlink_out_of_the_root_is_rejected(tmp_path):
    import os

    outside = tmp_path.parent / (tmp_path.name + "_outside")
    outside.mkdir()
    link = tmp_path / "link"
    os.symlink(outside, link)
    with pytest.raises(ValueError, match="outside the directories"):
        build_deepagent_options("deep_agent", str(link), None, True, None)


def test_empty_workspace_uses_first_allowed_root(tmp_path):
    options = build_deepagent_options("deep_agent", "", None, True, None)
    assert str(options["deepagent_backend"].cwd) == str(tmp_path.resolve())


def test_policy_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert deepagent_policy.deep_agent_enabled({}) is False
    assert deepagent_policy.deep_agent_enabled({deepagent_policy.ENABLE_ENV: "yes"}) is True
    assert deepagent_policy.allowed_roots({}) == (str(tmp_path.resolve()),)
    with pytest.raises(ValueError, match="NUL"):
        deepagent_policy.confine_directory("a\x00b", roots=[str(tmp_path)])


def test_shared_backend_helper_matches_cli_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace(None) == tmp_path.resolve()
    assert resolve_workspace("") == tmp_path.resolve()
    backend = create_host_shell_backend(None)
    assert str(backend.cwd) == str(tmp_path.resolve())


def test_shell_environment_follows_the_current_turn_directory(tmp_path, monkeypatch):
    from chemgraph.agent.deepagent_backend import update_shell_environment

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "chat"))
    backend = create_host_shell_backend(str(tmp_path))
    turn = tmp_path / "chat" / "turn_002_abcd"
    assert update_shell_environment(backend, CHEMGRAPH_LOG_DIR=str(turn)) is True
    assert backend.execute(_echo_env_command("CHEMGRAPH_LOG_DIR")).output.strip() == str(turn)
    with pytest.raises(ValueError, match="not an allowlisted"):
        update_shell_environment(backend, OPENAI_API_KEY="sk-x")
    assert update_shell_environment(object(), CHEMGRAPH_LOG_DIR="x") is False


def _echo_env_command(name):
    """Print environment variable *name* in the platform's host shell."""
    import os

    return f"echo %{name}%" if os.name == "nt" else f"echo ${name}"
