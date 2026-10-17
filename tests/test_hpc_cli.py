"""Configured HPC catalogs stay with the standalone DeepAgent workflow."""

import importlib
from types import SimpleNamespace

import pytest
import toml

from chemgraph.cli import commands
from chemgraph.registry.tools import ToolRegistry
from tests.test_hpc_batch import batch as batch


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


@pytest.mark.parametrize("workflow", ["main_agent", "single_agent"])
def test_hpc_initialization_validates_before_credentials(monkeypatch, workflow):
    monkeypatch.setattr(commands, "check_api_keys", lambda *_, **__: pytest.fail("must validate first"))
    with commands.console.capture() as output:
        assert commands.initialize_agent(
            "test", workflow, False, "state", False, 20, hpc_config={"targets": {}},
        ) is None
    assert "standalone deep_agent workflow" in " ".join(output.get().split())


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
    assert registry.get("hpc_list_targets").invoke({}) == {
        "targets": {}, "allow_unreviewed_mutations": False,
    }
    assert "run_ase" in registry.names()


def test_hpc_tools_compose_with_a_caller_registry(monkeypatch):
    received = {}
    monkeypatch.setattr(commands, "check_api_keys", lambda *_, **__: (True, ""))
    monkeypatch.setattr(commands, "_create_experimental_deepagent_backend", lambda *_, **__: None)
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.ChemGraph",
        lambda **kwargs: received.update(kwargs) or SimpleNamespace(**kwargs),
    )
    base = ToolRegistry().select(["calculator"])
    with commands.console.capture():
        commands.initialize_agent(
            "test", "deep_agent", False, "state", False, 20,
            hpc_config={"targets": {}}, deepagent_tool_registry=base,
        )
    registry = received["deepagent_tool_registry"]
    assert registry.names()[0] == "calculator" and "hpc_submit_job" in registry.names()
    assert "run_ase" not in registry.names()
    assert base.names() == ("calculator",)  # The caller's registry is not modified.


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("allow", [False, True])
def test_cli_bypass_with_hpc_requires_explicit_opt_in(tmp_path, monkeypatch, interactive, allow):
    config = tmp_path / "hpc.toml"
    config.write_text(toml.dumps({"hpc": {"targets": {}, "allow_unreviewed_mutations": allow}}))
    reached = []
    monkeypatch.setattr(cli, "interactive_mode", lambda **_: reached.append("interactive"))
    monkeypatch.setattr(cli, "initialize_agent", lambda *_, **__: reached.append("headless"))
    args = ["run", "-w", "deep_agent", "--config", str(config), "--dangerously-skip-approvals",
            "--deepagent-workspace", str(tmp_path)]
    args += ["--interactive"] if interactive else ["-q", "Submit the job"]
    with commands.console.capture() as output:
        try:
            cli._handle_run(cli.create_argument_parser().parse_args(args))
        except SystemExit as exc:
            reached.append(exc.code)
    text = " ".join(output.get().split())
    if not interactive and not allow:
        assert reached == [2] and "allow_unreviewed_mutations = true" in text
    else:
        assert 2 not in reached
        assert ("still require review" in text or "keeps reviews" in text) == (not allow)


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


def test_explicit_config_and_cli_restrictions(batch, tmp_path, monkeypatch):
    import importlib
    import toml

    cli = importlib.import_module("chemgraph.cli.main")
    service, root, *_ = batch
    config = tmp_path / "selected.toml"
    config.write_text(
        toml.dumps(
            {
                "general": {"workflow": "deep_agent"},
                "hpc": service.config.model_dump(exclude_none=True),
            }
        )
    )
    received = {}
    monkeypatch.setattr(
        cli, "interactive_mode", lambda **kwargs: received.update(kwargs)
    )
    args = cli.create_argument_parser().parse_args(
        [
            "run",
            "--interactive",
            "--config",
            str(config),
            "--tool",
            "hpc_list_targets",
        ]
    )
    cli._handle_run(args)
    registry = received["deepagent_tool_registry"]
    assert registry.names() == ("hpc_list_targets",)
    assert registry.get("hpc_list_targets").invoke({})["targets"]["polaris"][
        "local_root"
    ] == str(root.parent)
