"""Configured HPC catalogs stay with the standalone DeepAgent workflow."""

import importlib
from types import SimpleNamespace

import pytest
import toml

from chemgraph.cli import commands
from chemgraph.registry.tools import ToolRegistry


cli = importlib.import_module("chemgraph.cli.main")


@pytest.mark.parametrize("workflow", ["main_agent", "single_agent"])
def test_hpc_cli_rejects_unsupported_workflow(tmp_path, monkeypatch, workflow):
    config = tmp_path / "hpc.toml"
    config.write_text(toml.dumps({"hpc": {"targets": {}}}))
    monkeypatch.setattr(cli, "interactive_mode", lambda **_: pytest.fail("must validate first"))
    with commands.console.capture() as output, pytest.raises(SystemExit) as exc:
        cli._handle_run(cli.create_argument_parser().parse_args([
            "run", "--interactive", "-w", workflow, "--config", str(config),
        ]))
    assert exc.value.code == 2
    assert "use -w deep_agent" in " ".join(output.get().split())


@pytest.mark.parametrize("workflow", ["main_agent", "single_agent", "deep_agent"])
def test_hpc_initialization_validates_before_credentials(monkeypatch, workflow):
    monkeypatch.setattr(commands, "check_api_keys", lambda *_, **__: pytest.fail("must validate first"))
    options = {"hpc_config": {"targets": {}}}
    if workflow == "deep_agent":
        options["deepagent_tool_registry"] = ToolRegistry([])
    with commands.console.capture() as output:
        assert commands.initialize_agent("test", workflow, False, "state", False, 20, **options) is None
    expected = "not both" if workflow == "deep_agent" else "standalone deep_agent workflow"
    assert expected in " ".join(output.get().split())


def test_hpc_initialization_binds_standalone_catalog(monkeypatch):
    received = {}
    monkeypatch.setattr(commands, "check_api_keys", lambda *_, **__: (True, ""))
    monkeypatch.setattr(commands, "_create_experimental_deepagent_backend", lambda *_, **__: None)
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.ChemGraph",
        lambda **kwargs: received.update(kwargs) or SimpleNamespace(**kwargs),
    )
    with commands.console.capture():
        agent = commands.initialize_agent(
            "test", "deep_agent", False, "state", False, 20, hpc_config={"targets": {}},
        )
    assert agent is not None
    assert received["tool_registry"] is None
    registry = received["deepagent_tool_registry"]
    assert registry.get("hpc_list_targets").invoke({}) == {"targets": {}}


@pytest.mark.parametrize("startup", ["deep_agent", "main_agent"])
@pytest.mark.parametrize("names", [[], ["hpc_list_targets"]])
def test_hpc_catalog_is_isolated_across_workflow_switches(tmp_path, monkeypatch, startup, names):
    config = tmp_path / "hpc.toml"
    config.write_text(toml.dumps({
        "general": {"workflow": "deep_agent", "tools": names}, "hpc": {"targets": {}},
    }))
    replies = iter(["test-model", startup, "/workflow single_agent", "/workflow main_agent",
                    "/workflow deep_agent", "/quit"])
    calls = []
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_, **__: next(replies))
    monkeypatch.setattr(commands, "initialize_agent", lambda *args, **kwargs: (
        calls.append((args[1], kwargs)) or SimpleNamespace()
    ))
    monkeypatch.setattr(commands, "create_main_agent_session", lambda *_, **__: SimpleNamespace(thread_id="test"))
    with commands.console.capture():
        cli._handle_run(cli.create_argument_parser().parse_args([
            "run", "--interactive", "--config", str(config),
            "--checkpoint-db", str(tmp_path / "checkpoints.db"),
        ]))
    assert [workflow for workflow, _ in calls] == [startup, "single_agent", "main_agent", "deep_agent"]
    for workflow, options in calls:
        if workflow == "deep_agent":
            assert options["deepagent_tool_registry"].names() == tuple(names)
        else:
            assert options["deepagent_tool_registry"] is None
        if workflow == "main_agent":
            assert options["tool_registry"] is None
