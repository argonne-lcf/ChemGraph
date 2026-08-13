"""CLI tests for the long-lived main-agent workflow."""

import importlib
from types import SimpleNamespace

import pytest
import toml

from chemgraph.agent.main_session import MainAgentTurnResult, PendingInterrupt
from chemgraph.cli import commands
from chemgraph.cli.formatting import console

cli_main = importlib.import_module("chemgraph.cli.main")


@pytest.fixture(autouse=True)
def _isolate_durable_databases(monkeypatch, tmp_path):
    monkeypatch.setattr(
        commands,
        "DEFAULT_CHECKPOINT_DB",
        str(tmp_path / "checkpoints.db"),
    )
    monkeypatch.setattr(
        "chemgraph.memory.store.DEFAULT_DB_PATH",
        str(tmp_path / "sessions.db"),
    )


def _turn_result(*interrupts: PendingInterrupt) -> MainAgentTurnResult:
    return MainAgentTurnResult(
        thread_id="main-thread",
        status="waiting_for_user" if interrupts else "completed",
        assistant_response="turn complete",
        interrupts=interrupts,
        state={},
    )


class _FakeMainSession:
    def __init__(self, results, *, failed=False):
        self.thread_id = "main-thread"
        self.failed = failed
        self.results = iter(results)
        self.calls = []

    def _next_result(self):
        result = next(self.results)
        if isinstance(result, BaseException):
            self.failed = True
            raise result
        return result

    async def run(self, message):
        self.calls.append(("run", message))
        return self._next_result()

    async def resume(self, response):
        self.calls.append(("resume", response))
        return self._next_result()

    async def retry(self):
        self.calls.append(("retry", None))
        result = self._next_result()
        self.failed = False
        return result


def test_main_agent_is_a_cli_workflow():
    assert "main_agent" in commands.ALL_WORKFLOW_TYPES
    assert "main_agent" in cli_main._WORKFLOW_CHOICES


def test_main_agent_query_runs_each_turn_on_same_session():
    session = _FakeMainSession([_turn_result(), _turn_result()])

    first = commands.run_main_agent_query(session, "first")
    second = commands.run_main_agent_query(session, "second")

    assert first.assistant_response == "turn complete"
    assert second.assistant_response == "turn complete"
    assert session.calls == [("run", "first"), ("run", "second")]


def test_main_agent_query_answers_subagent_interrupt(monkeypatch):
    clarification = PendingInterrupt(
        id="worker-question",
        payload={"question": "Which calculator?"},
    )
    session = _FakeMainSession([_turn_result(clarification), _turn_result()])
    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: "EMT",
    )

    with console.capture() as capture:
        result = commands.run_main_agent_query(session, "calculate")

    assert result.assistant_response == "turn complete"
    assert session.calls == [("run", "calculate"), ("resume", "EMT")]
    assert "Which calculator?" in capture.get()


def test_main_agent_query_handles_deepagent_approval(monkeypatch):
    approval = PendingInterrupt(
        id="approval-id",
        payload={
            "action_requests": [
                {"name": "execute", "args": {"command": "pytest -q"}}
            ],
            "review_configs": [
                {
                    "action_name": "execute",
                    "allowed_decisions": ["approve", "reject"],
                }
            ],
        },
    )
    session = _FakeMainSession([_turn_result(approval), _turn_result()])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_args, **_kwargs: "approve")

    with console.capture() as capture:
        result = commands.run_main_agent_query(session, "run tests")

    assert result.assistant_response == "turn complete"
    assert session.calls == [
        ("run", "run tests"),
        ("resume", {"decisions": [{"type": "approve"}]}),
    ]
    assert "pytest -q" in capture.get()


def test_experimental_backend_requires_confirmation_and_filters_environment(
    monkeypatch,
    tmp_path,
):
    captured = {}

    class FakeBackend:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("deepagents.backends.LocalShellBackend", FakeBackend)
    monkeypatch.setattr(commands.Confirm, "ask", lambda *_args, **_kwargs: True)
    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    backend = commands._create_experimental_deepagent_backend(str(tmp_path))

    assert isinstance(backend, FakeBackend)
    assert captured["root_dir"] == tmp_path.resolve()
    assert captured["virtual_mode"] is True
    assert captured["inherit_env"] is False
    assert captured["env"]["PATH"] == "/test/bin"
    assert "OPENAI_API_KEY" not in captured["env"]


def test_experimental_backend_stops_when_confirmation_is_declined(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(commands.Confirm, "ask", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="was not approved"):
        commands._create_experimental_deepagent_backend(str(tmp_path))


def test_experimental_backend_rejects_missing_workspace(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        commands._create_experimental_deepagent_backend(
            str(tmp_path / "missing")
        )


def test_deepagent_approval_preserves_batched_action_order(monkeypatch):
    answers = iter(["approve", "reject"])
    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    payload = {
        "action_requests": [
            {"name": "write_file", "args": {"file_path": "/one"}},
            {"name": "delete", "args": {"file_path": "/two"}},
        ],
        "review_configs": [
            {
                "action_name": "write_file",
                "allowed_decisions": ["approve", "reject"],
            },
            {
                "action_name": "delete",
                "allowed_decisions": ["approve", "reject"],
            },
        ],
    }

    with console.capture():
        response = commands._prompt_for_interrupt(payload)

    assert response == {
        "decisions": [{"type": "approve"}, {"type": "reject"}]
    }


def test_deepagent_cli_boolean_flags():
    parser = cli_main.create_argument_parser()

    assert parser.parse_args(["--deepagent"]).deepagent is True
    assert parser.parse_args(["--no-deepagent"]).deepagent is False
    assert (
        parser.parse_args(["--checkpoint-db", "/tmp/checkpoints.db"]).checkpoint_db
        == "/tmp/checkpoints.db"
    )


def test_main_agent_query_failure_suggests_retry():
    session = _FakeMainSession([RuntimeError("temporary")])

    with console.capture() as capture:
        result = commands.run_main_agent_query(session, "calculate")

    assert result is None
    assert session.failed is True
    assert "`/retry` command" in capture.get()


def test_retry_main_agent_session_resumes_failed_operation():
    session = _FakeMainSession(
        [_turn_result()],
        failed=True,
    )

    result = commands.retry_main_agent_session(session)

    assert result.assistant_response == "turn complete"
    assert session.failed is False
    assert session.calls == [("retry", None)]


def test_interactive_main_agent_discards_session_on_quit(monkeypatch):
    session = _FakeMainSession([])
    agent = SimpleNamespace()
    answers = iter(["gpt-4o-mini", "main_agent", "calculate", "quit"])

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent, **_kwargs: session,
    )

    def fake_run(active_session, query, verbose=False, **_kwargs):
        assert active_session is session
        assert query == "calculate"
        assert verbose is False
        return _turn_result()

    monkeypatch.setattr(commands, "run_main_agent_query", fake_run)

    with console.capture():
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    assert session.calls == []


def test_interactive_main_agent_retry_command(monkeypatch):
    session = _FakeMainSession([], failed=True)
    agent = SimpleNamespace()
    answers = iter(["gpt-4o-mini", "main_agent", "/retry", "quit"])

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent, **_kwargs: session,
    )

    calls = []

    def fake_retry(active_session, verbose=False, **_kwargs):
        calls.append((active_session, verbose))
        active_session.failed = False
        return _turn_result()

    monkeypatch.setattr(commands, "retry_main_agent_session", fake_retry)

    with console.capture():
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    assert calls == [(session, False)]
    assert session.calls == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("/show AbC123", ("show", "AbC123")),
        ("/resume old-thread", ("resume", "old-thread")),
        ("/model Provider/Model-X", ("model", "Provider/Model-X")),
        ("/workflow single_agent", ("workflow", "single_agent")),
        ("/SHOW MixedCase", ("show", "MixedCase")),
        ("help", ("help", "")),
        ("retry", ("retry", "")),
        ("show me the tools", None),
        ("model a molecule", None),
    ],
)
def test_parse_interactive_input(value, expected):
    assert commands._parse_interactive_input(value) == expected


def test_interactive_show_prompt_reaches_main_agent(monkeypatch):
    prompt = "show me the list of all the tools your subagent has"
    session = _FakeMainSession([])
    agent = SimpleNamespace()
    answers = iter(["gpt-4o-mini", "main_agent", prompt, "quit"])
    calls = []

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent, **_kwargs: session,
    )
    monkeypatch.setattr(
        commands,
        "show_session",
        lambda _sid: pytest.fail("natural-language prompt called show_session"),
    )

    def fake_run(active_session, query, verbose=False, **_kwargs):
        calls.append((active_session, query, verbose))
        return _turn_result()

    monkeypatch.setattr(commands, "run_main_agent_query", fake_run)

    with console.capture():
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    assert calls == [(session, prompt, False)]


def test_interactive_slash_show_dispatches_to_session_command(monkeypatch):
    session = _FakeMainSession([])
    agent = SimpleNamespace()
    answers = iter(["gpt-4o-mini", "main_agent", "/show AbC123", "quit"])
    shown = []

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent, **_kwargs: session,
    )
    monkeypatch.setattr(commands, "show_session", shown.append)
    monkeypatch.setattr(
        commands,
        "run_main_agent_query",
        lambda *_args, **_kwargs: pytest.fail("/show reached the agent"),
    )

    with console.capture():
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    assert shown == ["AbC123"]


def test_interactive_slash_resume_dispatches_to_saved_session(monkeypatch):
    agent = SimpleNamespace(session_id="active-session")
    answers = iter(
        ["gpt-4o-mini", "single_agent", "/resume old-thread", "continue", "quit"]
    )
    calls = []

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)

    def fake_run(active_agent, query, verbose=False, resume_from=None):
        calls.append((active_agent, query, verbose, resume_from))
        return None

    monkeypatch.setattr(commands, "run_query", fake_run)

    with console.capture():
        commands.interactive_mode(generate_report=False)

    assert calls == [(agent, "continue", False, "old-thread")]


def test_interactive_slash_model_and_workflow_switches(monkeypatch):
    answers = iter(
        [
            "first-model",
            "main_agent",
            "/model Provider/Next-Model",
            "/workflow single_agent",
            "quit",
        ]
    )
    initial_agent = SimpleNamespace()
    next_agent = SimpleNamespace()
    single_agent = SimpleNamespace(session_id="single")
    agents = iter([initial_agent, next_agent, single_agent])
    initialization_calls = []

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )

    def fake_initialize(model, workflow, *_args, **_kwargs):
        initialization_calls.append((model, workflow))
        return next(agents)

    monkeypatch.setattr(commands, "initialize_agent", fake_initialize)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent, **_kwargs: SimpleNamespace(thread_id="main", failed=False),
    )

    with console.capture():
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    assert initialization_calls == [
        ("first-model", "main_agent"),
        ("Provider/Next-Model", "main_agent"),
        ("Provider/Next-Model", "single_agent"),
    ]


def test_interactive_deepagent_setting_survives_workflow_switches(monkeypatch):
    answers = iter(
        [
            "first-model",
            "main_agent",
            "/workflow single_agent",
            "/workflow main_agent",
            "quit",
        ]
    )
    agents = iter(
        [
            SimpleNamespace(),
            SimpleNamespace(session_id="single"),
            SimpleNamespace(),
        ]
    )
    initialization_calls = []
    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )

    def fake_initialize(*_args, **kwargs):
        initialization_calls.append(
            (kwargs["enable_deepagent"], kwargs["deepagent_workspace"])
        )
        return next(agents)

    monkeypatch.setattr(commands, "initialize_agent", fake_initialize)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent, **_kwargs: SimpleNamespace(thread_id="main", failed=False),
    )

    with console.capture():
        commands.interactive_mode(
            workflow="main_agent",
            generate_report=False,
            enable_deepagent=True,
            deepagent_workspace="/workspace",
        )

    assert initialization_calls == [
        (True, "/workspace"),
        (False, None),
        (True, "/workspace"),
    ]


def test_interactive_reports_invalid_slash_commands(monkeypatch):
    agent = SimpleNamespace()
    session = _FakeMainSession([])
    answers = iter(
        ["gpt-4o-mini", "main_agent", "/show", "/not-a-command", "quit"]
    )

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent, **_kwargs: session,
    )

    with console.capture() as capture:
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    output = capture.get()
    assert "Usage: /show <session_id>" in output
    assert "Unknown interactive command: /not-a-command" in output
    assert "Type /help" in output


def _run_args(**overrides):
    values = {
        "list_models": False,
        "check_keys": False,
        "list_sessions": False,
        "show_session": None,
        "delete_session": None,
        "config": None,
        "verbose": 0,
        "base_url": None,
        "model": "gpt-4o-mini",
        "workflow": "main_agent",
        "resume": None,
        "interactive": False,
        "structured": False,
        "output": "state",
        "report": False,
        "human_supervised": False,
        "recursion_limit": 20,
        "deepagent": None,
        "deepagent_workspace": None,
        "mcp_url": None,
        "mcp_command": None,
        "mcp_server_name": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_main_agent_requires_interactive_cli_mode():
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        cli_main._handle_run(_run_args())

    assert exc_info.value.code == 2
    assert "requires interactive mode" in capture.get()


def test_main_agent_dispatches_persistent_resume(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "interactive_mode",
        lambda **kwargs: captured.update(kwargs),
    )

    cli_main._handle_run(_run_args(interactive=True, resume="old-thread"))

    assert captured["resume_session"] == "old-thread"


@pytest.mark.parametrize(
    ("cli_value", "expected"),
    [(None, True), (False, False)],
)
def test_deepagent_toml_and_cli_precedence(
    monkeypatch,
    tmp_path,
    cli_value,
    expected,
):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps(
            {
                "general": {
                    "enable_deepagent": True,
                    "deepagent_workspace": str(tmp_path),
                }
            }
        )
    )
    captured = {}
    monkeypatch.setattr(
        cli_main,
        "interactive_mode",
        lambda **kwargs: captured.update(kwargs),
    )

    cli_main._handle_run(
        _run_args(
            config=str(config_path),
            interactive=True,
            deepagent=cli_value,
        )
    )

    assert captured["enable_deepagent"] is expected
    assert captured["deepagent_workspace"] == (
        str(tmp_path) if expected else None
    )
