"""CLI configuration and error-handling regressions for Deep Agent."""

import io
import os
import select
import subprocess
import sys
import time
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
@pytest.mark.parametrize("missing", [False, True])
def test_saved_deepagent_settings_do_not_block_other_workflows(
    tmp_path, dispatch, interactive, enabled, missing
):
    skill_dir = tmp_path / "moved-away" if missing else tmp_path
    path = tmp_path / "config.toml"
    path.write_text(
        toml.dumps(
            {
                "general": {
                    "workflow": "single_agent",
                    "enable_deepagent": enabled,
                    "deepagent_workspace": str(tmp_path),
                    "deepagent_skills": [str(skill_dir)],
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
    assert dispatch["deepagent_skill_dirs"] == (
        (str(skill_dir),) if interactive else None
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


@pytest.mark.parametrize("subcommand", [[], ["run"]])
@pytest.mark.parametrize("flag_style", ["separate", "equals"])
@pytest.mark.parametrize(
    "config_limit,flag,expected",
    [(None, None, 200), ({}, None, 200), (17, None, 17),
     (None, 33, 33), ({}, 33, 33), (17, 33, 33), (17, 200, 200)],
)
def test_recursion_limit_defaults_and_overrides(
    tmp_path, monkeypatch, dispatch, subcommand, flag_style, config_limit, flag, expected
):
    argv = [*subcommand, "--interactive"]
    if config_limit is not None:
        path = tmp_path / "config.toml"
        general = {} if config_limit == {} else {"recursion_limit": config_limit}
        path.write_text(toml.dumps({"general": general}))
        argv += ["--config", str(path)]
    if flag is not None:
        argv += (["--recursion-limit", str(flag)] if flag_style == "separate"
                 else [f"--recursion-limit={flag}"])
    # Programmatic parsing must work independently of the process's argv.
    monkeypatch.setattr(sys, "argv", ["chemgraph"])
    cli_main._handle_run(cli_main.create_argument_parser().parse_args(argv))
    assert dispatch["recursion_limit"] == expected


@pytest.mark.parametrize(
    "flags",
    [
        ["--deepagent"],
        ["--deepagent-workspace", "/tmp"],
        ["--deepagent-skill", "/skills/"],
        ["--deepagent-discover-skills"],
        ["--no-deepagent-discover-skills"],
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


def _review_terminal(monkeypatch, text):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(text))
    monkeypatch.setattr(commands, "console", Console(file=output, width=120))
    return output


@pytest.mark.parametrize("answer", ["", "  ", "1", "y", "YES", "a", " Approve "])
def test_review_approves_from_terminal_input(monkeypatch, answer):
    output = _review_terminal(monkeypatch, answer + "\n")
    assert commands._prompt_for_interrupt(_review(["approve", "reject"])) == {
        "decisions": [{"type": "approve"}]
    }
    assert "Enter / y" in output.getvalue()


@pytest.mark.parametrize("answer", ["2", "n", "NO", "r", " Reject "])
def test_review_rejects_from_terminal_input(monkeypatch, answer):
    _review_terminal(monkeypatch, answer + "\n")
    assert commands._prompt_for_interrupt(_review(["approve", "reject"])) == {
        "decisions": [{"type": "reject"}]
    }


def test_review_feedback_preserves_text(monkeypatch):
    _review_terminal(monkeypatch, "  Use EMT  instead of [red]MACE[/red].  \n")
    assert commands._prompt_for_interrupt(_review(["approve", "reject"])) == {
        "decisions": [{"type": "reject", "message": "Use EMT  instead of [red]MACE[/red]."}]
    }


@pytest.mark.parametrize("answer", ["v", "VIEW"])
def test_full_view_returns_to_same_action_without_approving(monkeypatch, answer):
    from contextlib import contextmanager

    output = _review_terminal(monkeypatch, f"{answer}\nUse EMT\n")
    payload = _review(["approve", "reject"])
    payload["action_requests"][0] = {"name": "write_file", "args": {
        "file_path": "/large.txt", "content": "before\n" * 50 + "middle\x1b[2K\n" + "after\n" * 50,
    }}
    payload["review_configs"][0]["action_name"] = "write_file"
    pages = []
    drains = []

    @contextmanager
    def pager(**kwargs):
        assert kwargs == {"styles": False, "links": False}
        with commands.console.capture() as capture:
            yield
        pages.append(capture.get())

    monkeypatch.setattr(commands.console, "pager", pager)
    monkeypatch.setattr(commands, "_clear_pending_review_input", lambda: drains.append(True))
    result = commands._prompt_for_interrupt(payload)
    assert result == {"decisions": [{"type": "reject", "message": "Use EMT"}]}
    assert len(pages) == 1 and "middle\\x1b[2K" in pages[0]
    assert "\x1b" not in pages[0]
    assert len(drains) == 2
    assert output.getvalue().count("Tool: write_file | Path: /large.txt") == 2


def test_view_word_remains_feedback_for_complete_preview(monkeypatch):
    _review_terminal(monkeypatch, "view\n")
    assert commands._prompt_for_interrupt(_review(["approve", "reject"])) == {
        "decisions": [{"type": "reject", "message": "view"}]
    }


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_review_input_drain_is_tty_only_and_portable(monkeypatch, platform):
    calls = []
    tty = SimpleNamespace(isatty=lambda: True, fileno=lambda: 123)
    monkeypatch.setattr(sys, "stdin", tty)
    monkeypatch.setattr(sys, "platform", platform)
    keys = iter([True, True, False])
    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(
        kbhit=lambda: next(keys), getwch=lambda: calls.append("key"),
    ))
    monkeypatch.setitem(sys.modules, "termios", SimpleNamespace(
        tcflush=lambda fd, mode: calls.append((fd, mode)), TCIFLUSH=0, error=RuntimeError,
    ))
    commands._clear_pending_review_input()
    assert calls == (["key", "key"] if platform == "win32" else [(123, 0)])
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n"))
    calls.clear()
    commands._clear_pending_review_input()
    assert not calls and sys.stdin.read() == "\n"


@pytest.mark.parametrize("error", [OSError, ValueError, RuntimeError])
def test_review_input_drain_tolerates_unavailable_terminal(monkeypatch, error):
    def fail(*_):
        raise error("unavailable")

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True, fileno=lambda: 123))
    monkeypatch.setitem(sys.modules, "termios", SimpleNamespace(
        tcflush=fail, TCIFLUSH=0, error=RuntimeError,
    ))
    commands._clear_pending_review_input()


@pytest.mark.skipif(os.name != "posix", reason="requires a POSIX pseudo-terminal")
def test_tty_queued_enters_cannot_approve_next_action():
    import pty

    script = """
from chemgraph.cli import commands
from rich.console import Console
commands.console = Console(color_system=None, width=100)
payload = {
    'action_requests': [{'name': 'execute', 'args': {'command': 'echo test'}}] * 2,
    'review_configs': [{'action_name': 'execute', 'allowed_decisions': ['approve', 'reject']}],
}
input('START> ')
print('RESULT', commands._prompt_for_interrupt(payload), flush=True)
"""
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", script], stdin=slave, stdout=slave, stderr=slave,
    )
    os.close(slave)
    pending = b""

    def read_until(marker):
        nonlocal pending
        deadline = time.monotonic() + 30
        while marker not in pending:
            remaining = deadline - time.monotonic()
            assert remaining > 0, pending.decode(errors="replace")
            assert select.select([master], [], [], remaining)[0], pending.decode(errors="replace")
            pending += os.read(master, 65536)
        prefix, pending = pending.split(marker, 1)
        return prefix

    try:
        read_until(b"START> ")
        os.write(master, b"\n\n")  # Input queued before the first review.
        read_until(b"Decision (approve): ")
        os.write(master, b"\n\n")  # Approve once and accidentally double-tap Enter.
        read_until(b"Decision (approve): ")
        os.write(master, b"n\n")
        read_until(b"RESULT ")
        result = read_until(b"\r\n")
        assert result == b"{'decisions': [{'type': 'approve'}, {'type': 'reject'}]}"
        assert process.wait(timeout=10) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master)


@pytest.mark.parametrize("allowed,text,expected", [
    (["approve"], "n\nUse EMT\n\n", {"type": "approve"}),
    (["reject"], "y\n\n", {"type": "reject"}),
    (["reject"], "y\nUse EMT\n", {"type": "reject", "message": "Use EMT"}),
])
def test_review_respects_restricted_policy(monkeypatch, allowed, text, expected):
    output = _review_terminal(monkeypatch, text)
    assert commands._prompt_for_interrupt(_review(allowed)) == {"decisions": [expected]}
    assert "not allowed" in output.getvalue()
    assert ("1. Approve" in output.getvalue()) == ("approve" in allowed)
    assert ("skip this action" in output.getvalue()) == ("reject" in allowed)


def test_review_batches_keep_order_and_interrupt_ids(monkeypatch):
    output = _review_terminal(monkeypatch, "\nn\nUse EMT\ny\n")
    batch = _review(["approve", "reject"])
    batch["action_requests"] *= 3
    session = _FakeMainSession([
        _turn_result(
            PendingInterrupt("batch", batch),
            PendingInterrupt("other", _review(["approve"])),
        ),
        _turn_result(),
    ])
    assert commands.run_main_agent_query(session, "test") is not None
    assert session.calls[-1] == ("resume", {
        "batch": {"decisions": [
            {"type": "approve"}, {"type": "reject"},
            {"type": "reject", "message": "Use EMT"},
        ]},
        "other": {"decisions": [{"type": "approve"}]},
    })
    assert "Review action 3 of 3" in output.getvalue()


@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt])
def test_cancelled_review_never_resumes(monkeypatch, error):
    _review_terminal(monkeypatch, "")

    def cancel(*args, **kwargs):
        raise error

    monkeypatch.setattr("builtins.input", cancel)
    session = _FakeMainSession([
        _turn_result(PendingInterrupt("review", _review(["approve", "reject"]))),
    ])
    if error is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            commands.run_main_agent_query(session, "test")
    else:
        assert commands.run_main_agent_query(session, "test") is None
    assert session.calls == [("run", "test")]


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


@pytest.mark.parametrize(
    "configured,flag,expected",
    [(True, None, True), (False, None, False),
     (True, "--no-deepagent-discover-skills", False),
     (False, "--deepagent-discover-skills", True)],
)
def test_skill_discovery_cli_overrides_toml(tmp_path, dispatch, configured, flag, expected):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"general": {
        "workflow": "deep_agent", "deepagent_discover_skills": configured,
        "deepagent_skills": [str(tmp_path)],
    }}))
    argv = ["run", "--interactive", "--config", str(path)]
    if flag:
        argv.append(flag)
    cli_main._handle_run(cli_main.create_argument_parser().parse_args(argv))
    assert dispatch["deepagent_discover_skills"] is expected
    assert dispatch["deepagent_skill_dirs"] == (str(tmp_path.resolve()),)


def test_skill_discovery_rejects_non_boolean_toml(tmp_path, dispatch):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"general": {
        "workflow": "deep_agent", "deepagent_discover_skills": "false",
    }}))
    args = cli_main.create_argument_parser().parse_args(["run", "--interactive", "--config", str(path)])
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(args)
    assert exc.value.code == 2
    assert not dispatch


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_host_paths_use_invocation_directory_and_override_toml(
    monkeypatch, tmp_path, dispatch, explicit
):
    workspace = tmp_path / "work"
    workspace.mkdir()
    external = tmp_path / "external skills"
    external.mkdir()
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    config = config_dir / "settings.toml"
    config.write_text(toml.dumps({"general": {
        "workflow": "deep_agent", "deepagent_workspace": str(workspace),
        "deepagent_skills": ["./external skills"],
    }}))
    monkeypatch.chdir(tmp_path)
    argv = ["run", "--interactive", "--config", str(config)]
    if explicit:
        argv += ["--deepagent-skill", "configuration", "--deepagent-skill", "./work"]
    cli_main._handle_run(cli_main.create_argument_parser().parse_args(argv))
    expected = (str(config_dir), str(workspace)) if explicit else (str(external),)
    assert dispatch["deepagent_skill_dirs"] == expected
    assert "deepagent_skills" not in dispatch


@pytest.mark.parametrize("name", ["missing", "[missing][/red]"])
def test_invalid_cli_host_path_fails_before_initialization(tmp_path, dispatch, name):
    args = cli_main.create_argument_parser().parse_args([
        "run", "-w", "deep_agent", "--deepagent-skill",
        (tmp_path / name).as_posix(),
    ])
    with commands.console.capture() as capture, pytest.raises(SystemExit) as exc:
        cli_main._handle_run(args)
    assert exc.value.code == 2
    output = capture.get()
    assert "Cannot access skill directory" in output
    assert name in output.replace("\n", "")
    assert not dispatch


@pytest.mark.parametrize("interactive", [False, True])
@pytest.mark.parametrize("override", [False, True])
def test_local_catalog_cli_precedence_is_lazy(tmp_path, dispatch, monkeypatch, interactive, override):
    from chemgraph.registry import ToolRegistry

    monkeypatch.setattr(ToolRegistry, "get", lambda *_a, **_k: pytest.fail("must not import tools"))
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"general": {
        "workflow": "deep_agent", "tools": ["file_to_atomsdata"],
    }}))
    argv = ["run", "--config", str(path)]
    argv += ["--interactive"] if interactive else [
        "-q", "test", "--deepagent-workspace", str(tmp_path),
        "--deepagent-dangerously-skip-approvals",
    ]
    if override:
        argv += ["--tool", "smiles_to_coordinate_file", "--tool", "smiles_to_coordinate_file"]
    cli_main._handle_run(cli_main.create_argument_parser().parse_args(argv))
    assert dispatch["deepagent_tool_registry"].names() == (
        ("smiles_to_coordinate_file",) if override else ("file_to_atomsdata",)
    )


@pytest.mark.parametrize("value", ["run_ase", "", False, 0, [""], [123], ["unknown"]])
def test_invalid_local_catalog_fails_before_initialization(tmp_path, dispatch, value):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"general": {"workflow": "deep_agent", "tools": value}}))
    args = cli_main.create_argument_parser().parse_args(["run", "--interactive", "--config", str(path)])
    with commands.console.capture() as capture, pytest.raises(SystemExit) as exc:
        cli_main._handle_run(args)
    assert exc.value.code == 2
    assert "Invalid local tools" in capture.get()
    assert not dispatch


def test_explicit_local_catalog_requires_standalone_deep_agent(dispatch):
    args = cli_main.create_argument_parser().parse_args([
        "run", "--interactive", "-w", "main_agent", "--tool", "file_to_atomsdata",
    ])
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(args)
    assert exc.value.code == 2
    assert not dispatch


def test_omitted_catalog_selects_python_default(dispatch):
    cli_main._handle_run(cli_main.create_argument_parser().parse_args([
        "run", "--interactive", "-w", "deep_agent",
    ]))
    assert dispatch["deepagent_tool_registry"] is None


@pytest.mark.parametrize("explicit_empty", [False, True])
def test_codex_command_defaults_to_discovery_with_skills_disabled(
    monkeypatch, tmp_path, dispatch, explicit_empty,
):
    from chemgraph.registry import ToolRegistry
    from chemgraph.models.endpoints import PreparedModel
    from tests.test_registry_middleware import CatalogModel, names
    from langchain_core.messages import AIMessage, HumanMessage

    monkeypatch.chdir(tmp_path)
    if explicit_empty:
        (tmp_path / "config.toml").write_text("[general]\ntools = []\n")
    model = CatalogModel(responses=[AIMessage(content="Ready")])
    monkeypatch.setattr(ToolRegistry, "get", lambda *_a, **_k: pytest.fail("must stay lazy"))
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.load_chat_model_prepared",
        lambda **_: (model, PreparedModel(endpoint_name="test", protocol="openai_compatible", client_kwargs={})),
    )
    argv = [
        "run", "--interactive", "--model", "codex:gpt-5.6-sol",
        "--deepagent-workspace", ".", "--no-deepagent-discover-skills",
        "--workflow", "deepagent",
    ]
    if explicit_empty:
        argv += ["--config", str(tmp_path / "config.toml")]
    args = cli_main.create_argument_parser().parse_args(argv)
    cli_main._handle_run(args)
    assert dispatch["workflow"] == "deep_agent"
    assert dispatch["deepagent_discover_skills"] is False
    agent = ChemGraph(
        model_name=dispatch["model"], workflow_type=dispatch["workflow"],
        deepagent_tool_registry=dispatch["deepagent_tool_registry"],
        deepagent_discover_skills=dispatch["deepagent_discover_skills"],
        enable_memory=False, log_dir=str(tmp_path),
    )
    agent.workflow.invoke(
        {"messages": [HumanMessage(content="Ready?")]},
        {"configurable": {"thread_id": "default-discovery"}},
    )
    expected = () if explicit_empty else tuple(
        spec.name for spec in ToolRegistry().specs() if not spec.interactive
    )
    assert agent.deepagent_tool_registry.names() == expected
    assert agent.deepagent_tool_registry._tools == {}
    discovery = set() if explicit_empty else {"search_tools", "load_tools"}
    assert names(model.schemas[0]) & {"search_tools", "load_tools"} == discovery
    assert not set(ToolRegistry().names()) & names(model.schemas[0])


@pytest.mark.parametrize("enabled", [False, True])
def test_initialization_reports_catalog_status(monkeypatch, enabled):
    from chemgraph.registry import ToolRegistry

    registry = ToolRegistry() if enabled else ToolRegistry([])
    monkeypatch.setattr(commands, "check_api_keys", lambda *_a, **_k: (True, ""))
    monkeypatch.setattr(commands, "_create_experimental_deepagent_backend", lambda *_a, **_k: None)
    monkeypatch.setattr(commands.time, "sleep", lambda *_: None)
    monkeypatch.setattr(
        "chemgraph.agent.llm_agent.ChemGraph",
        lambda **_: SimpleNamespace(deepagent_tool_registry=registry),
    )
    with commands.console.capture() as capture:
        agent = commands.initialize_agent("fake", "deep_agent", False, "state", False, 20)
    assert agent is not None
    expected = f"{len(registry.names())} discoverable" if enabled else "discovery disabled"
    assert expected in capture.get()


@pytest.mark.parametrize("catalog_mode", ["default", "empty", "restricted"])
def test_interactive_catalog_survives_model_and_workflow_changes(monkeypatch, catalog_mode):
    from chemgraph.registry import ToolRegistry

    registry = (
        None if catalog_mode == "default" else ToolRegistry([])
        if catalog_mode == "empty" else ToolRegistry([ToolRegistry().get_spec("calculator")])
    )
    initialized = []
    prompts = iter(["initial-model", "deep_agent", "/model another-model", "/workflow single_agent", "/workflow deep_agent", "/quit"])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_a, **_k: next(prompts))
    monkeypatch.setattr(commands, "create_banner", lambda: None)
    monkeypatch.setattr(commands, "initialize_agent", lambda *args, **kwargs: (
        initialized.append((args[1], kwargs.get("deepagent_tool_registry"))) or SimpleNamespace()
    ))
    commands.interactive_mode(workflow="deep_agent", deepagent_tool_registry=registry)
    assert initialized == [
        ("deep_agent", registry), ("deep_agent", registry),
        ("single_agent", None), ("deep_agent", registry),
    ]


@pytest.mark.parametrize("restriction", [None, [], ["calculator"], ["ask_human"]])
@pytest.mark.parametrize("startup_workflow", ["single_agent", "deep_agent"])
def test_cli_retains_catalog_before_selecting_deep_agent(
    monkeypatch, tmp_path, restriction, startup_workflow,
):
    from chemgraph.registry import ToolRegistry

    settings = {"workflow": "single_agent"}
    if restriction is not None:
        settings["tools"] = restriction
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"general": settings}))
    prompts = iter([
        "initial-model", startup_workflow, "/workflow deep_agent",
        "/model another-model", "/workflow single_agent", "/workflow deep_agent", "/quit",
    ])
    initialized = []
    monkeypatch.setattr(ToolRegistry, "get", lambda *_a, **_k: pytest.fail("must stay lazy"))
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_a, **_k: next(prompts))
    monkeypatch.setattr(commands, "create_banner", lambda: None)
    monkeypatch.setattr(commands, "initialize_agent", lambda *args, **kwargs: (
        initialized.append((args[1], kwargs["deepagent_tool_registry"])) or SimpleNamespace()
    ))
    cli_main._handle_run(cli_main.create_argument_parser().parse_args([
        "run", "--interactive", "--config", str(path),
    ]))
    assert [workflow for workflow, _ in initialized] == [
        startup_workflow, "deep_agent", "deep_agent", "single_agent", "deep_agent",
    ]
    for workflow, registry in initialized:
        if workflow != "deep_agent" or restriction is None:
            assert registry is None
        else:
            assert registry.names() == tuple(restriction)
    assert initialized[1][1] is initialized[-1][1]


def test_noninteractive_other_workflow_ignores_catalog(tmp_path, dispatch):
    path = tmp_path / "config.toml"
    path.write_text(toml.dumps({"general": {
        "workflow": "single_agent", "tools": ["unknown"],
    }}))
    cli_main._handle_run(cli_main.create_argument_parser().parse_args([
        "run", "-q", "test", "--config", str(path),
    ]))
    assert dispatch["deepagent_tool_registry"] is None
