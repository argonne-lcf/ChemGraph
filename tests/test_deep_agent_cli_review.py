"""CLI configuration and error-handling regressions for Deep Agent."""

from types import SimpleNamespace

import pytest
import toml
from rich.console import Console

from chemgraph.agent.interrupts import PendingInterrupt
from chemgraph.agent.llm_agent import ChemGraph, HumanInputRequired
from chemgraph.cli import commands
from tests.test_main_agent_cli import _FakeMainSession, _turn_result, cli_main


@pytest.fixture
def dispatch(monkeypatch):
    received = {}
    monkeypatch.setattr(
        cli_main, "interactive_mode", lambda **kwargs: received.update(kwargs)
    )
    monkeypatch.setattr(
        cli_main,
        "initialize_agent",
        lambda *args, **kwargs: (
            received.update(workflow=args[1], **kwargs) or SimpleNamespace()
        ),
    )
    monkeypatch.setattr(cli_main, "run_query", lambda *args, **kwargs: "done")
    monkeypatch.setattr(cli_main, "format_response", lambda *args, **kwargs: None)
    return received


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_saved_deepagent_settings_do_not_block_other_workflows(
    tmp_path, dispatch, interactive, enabled
):
    path = tmp_path / "config.toml"
    path.write_text(
        toml.dumps(
            {
                "general": {
                    "workflow": "single_agent",
                    "enable_deepagent": enabled,
                    "deepagent_workspace": str(tmp_path),
                    "deepagent_skills": ["/workspace/skills/"],
                }
            }
        )
    )
    argv = ["run", "--config", str(path), "-q", "test"]
    if interactive:
        argv.append("--interactive")
    cli_main._handle_run(cli_main.create_argument_parser().parse_args(argv))
    assert dispatch["workflow"] == "single_agent"
    assert dispatch["deepagent_workspace"] == (str(tmp_path) if interactive else None)
    assert dispatch["deepagent_skills"] == (
        ["/workspace/skills/"] if interactive else None
    )
    if interactive:
        assert dispatch["enable_deepagent"] is enabled


@pytest.mark.parametrize(
    "override,expected", [(None, "deep_agent"), ("single_agent", "single_agent")]
)
def test_workflow_config_and_explicit_cli_precedence(
    tmp_path, dispatch, override, expected
):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"general": {"workflow": "deepagent"}}))
    argv = ["run", "--interactive", "--config", str(path)]
    if override:
        argv += ["-w", override]
    cli_main._handle_run(cli_main.create_argument_parser().parse_args(argv))
    assert dispatch["workflow"] == expected


def test_missing_workflow_defaults_to_single_agent(dispatch):
    cli_main._handle_run(
        cli_main.create_argument_parser().parse_args(["run", "--interactive"])
    )
    assert dispatch["workflow"] == "single_agent"


@pytest.mark.parametrize(
    "flags",
    [
        ["--deepagent"],
        ["--deepagent-workspace", "/tmp"],
        ["--deepagent-skill", "/skills/"],
    ],
)
def test_explicit_incompatible_deepagent_flags_still_fail(dispatch, flags):
    args = cli_main.create_argument_parser().parse_args(
        ["run", "--interactive", "-w", "single_agent", *flags]
    )
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(args)
    assert exc.value.code == 2
    assert not dispatch


@pytest.mark.parametrize("skills", ["/workspace/skills/", [""], [123]])
def test_malformed_active_skills_are_cli_input_errors(tmp_path, dispatch, skills):
    path = tmp_path / "config.toml"
    path.write_text(
        toml.dumps({"general": {"workflow": "deep_agent", "deepagent_skills": skills}})
    )
    args = cli_main.create_argument_parser().parse_args(
        ["run", "--interactive", "--config", str(path)]
    )
    with commands.console.capture() as capture, pytest.raises(SystemExit) as exc:
        cli_main._handle_run(args)
    assert exc.value.code == 2
    assert "Invalid Deep Agent skills" in capture.get()
    assert not dispatch


def test_interactive_initialization_reports_invalid_skills(monkeypatch):
    monkeypatch.setattr(
        commands,
        "_create_experimental_deepagent_backend",
        lambda *_args, **_kwargs: pytest.fail("must validate before enabling shell"),
    )
    with commands.console.capture() as capture:
        result = commands.initialize_agent(
            "fake", "deep_agent", False, "state", False, 20, deepagent_skills=[""]
        )
    assert result is None
    assert "must not be empty" in capture.get()
    with pytest.raises(ValueError, match="requires enable_deepagent"):
        ChemGraph(
            workflow_type="single_agent",
            deepagent_skills="/skills/",
            enable_memory=False,
        )


@pytest.mark.parametrize("approve", [False, True])
def test_cwd_workspace_is_named_and_confirmation_is_required(
    monkeypatch, tmp_path, approve
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(commands, "console", Console(width=300))
    confirmations = []
    created = []
    monkeypatch.setattr(
        commands.Confirm,
        "ask",
        lambda *args, **kwargs: confirmations.append(kwargs) or approve,
    )
    monkeypatch.setattr(
        "deepagents.backends.LocalShellBackend",
        lambda **kwargs: created.append(kwargs) or kwargs,
    )
    with commands.console.capture() as capture:
        if approve:
            commands._create_experimental_deepagent_backend(None)
        else:
            with pytest.raises(RuntimeError, match="not approved"):
                commands._create_experimental_deepagent_backend(None)
    assert str(tmp_path.resolve()) in capture.get()
    assert confirmations == [{"default": False}]
    assert bool(created) is approve


def _review(allowed):
    return {
        "action_requests": [{"name": "execute", "args": {"command": "test"}}],
        "review_configs": [{"action_name": "execute", "allowed_decisions": allowed}],
    }


@pytest.mark.parametrize("main_agent", [False, True])
@pytest.mark.parametrize(
    "payload", [_review(["edit"]), {"action_requests": [None], "review_configs": []}]
)
def test_prompt_errors_are_reported_without_tracebacks(
    monkeypatch, main_agent, payload
):
    monkeypatch.setattr(commands.time, "sleep", lambda _: None)
    pending = PendingInterrupt(id="review", payload=payload)
    if main_agent:
        session = _FakeMainSession([_turn_result(pending)])

        def run():
            return commands.run_main_agent_query(session, "test")
    else:

        class Agent:
            async def run(self, *args, **kwargs):
                raise HumanInputRequired("review", interrupts=(pending,))

        def run():
            return commands.run_query(Agent(), "test")

    with commands.console.capture() as capture:
        assert run() is None
    assert "Error" in capture.get()


@pytest.mark.parametrize("approval", [False, True])
def test_main_agent_limit_counts_questions_only(monkeypatch, approval):
    payload = _review(["approve", "reject"]) if approval else {"question": "Continue?"}
    pending = [PendingInterrupt(id=str(i), payload=payload) for i in range(12)]
    session = _FakeMainSession([*(_turn_result(p) for p in pending), _turn_result()])
    monkeypatch.setattr(
        commands,
        "_prompt_for_interrupt",
        lambda _: {"decisions": [{"type": "approve"}]} if approval else "yes",
    )
    result = commands.run_main_agent_query(session, "test")
    assert (result is not None) is approval
    assert len(session.calls) == (13 if approval else 11)
