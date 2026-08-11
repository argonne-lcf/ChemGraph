"""CLI tests for the long-lived main-agent workflow."""

import importlib
from types import SimpleNamespace

import pytest

from chemgraph.agent.main_session import MainAgentTurnResult, PendingInterrupt
from chemgraph.cli import commands
from chemgraph.cli.formatting import console

cli_main = importlib.import_module("chemgraph.cli.main")


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


def test_main_agent_query_failure_suggests_retry():
    session = _FakeMainSession([RuntimeError("temporary")])

    with console.capture() as capture:
        result = commands.run_main_agent_query(session, "calculate")

    assert result is None
    assert session.failed is True
    assert "`retry` command" in capture.get()


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
        lambda _agent: session,
    )

    def fake_run(active_session, query, verbose=False):
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
    answers = iter(["gpt-4o-mini", "main_agent", "retry", "quit"])

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda _agent: session,
    )

    calls = []

    def fake_retry(active_session, verbose=False):
        calls.append((active_session, verbose))
        active_session.failed = False
        return _turn_result()

    monkeypatch.setattr(commands, "retry_main_agent_session", fake_retry)

    with console.capture():
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    assert calls == [(session, False)]
    assert session.calls == []


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
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_main_agent_requires_interactive_cli_mode():
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        cli_main._handle_run(_run_args())

    assert exc_info.value.code == 2
    assert "requires interactive mode" in capture.get()


def test_main_agent_rejects_persistent_resume():
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        cli_main._handle_run(_run_args(interactive=True, resume="old-thread"))

    assert exc_info.value.code == 2
    assert "--resume is not supported" in capture.get()
