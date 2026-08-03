"""Tests for the `chemgraph resume` CLI subcommand.

Hermetic: parser wiring + dispatch only. `_handle_run` is monkeypatched so no
agent/LLM is created.
"""

import importlib
import sys

# NOTE: `chemgraph.cli.__init__` does `from .main import main`, which shadows the
# submodule name `chemgraph.cli.main` with the function. Load the real module via
# sys.modules to reach create_argument_parser / _handle_resume.
importlib.import_module("chemgraph.cli.main")
cli_main = sys.modules["chemgraph.cli.main"]


class TestResumeParser:
    def test_parses_positional_and_query(self):
        parser = cli_main.create_argument_parser()
        args = parser.parse_args(["resume", "abc123", "-q", "next step"])
        assert args.command == "resume"
        assert args.session_id == "abc123"
        assert args.query == "next step"

    def test_query_optional(self):
        parser = cli_main.create_argument_parser()
        args = parser.parse_args(["resume", "abc123"])
        assert args.session_id == "abc123"
        assert args.query is None

    def test_inherits_run_args(self):
        parser = cli_main.create_argument_parser()
        args = parser.parse_args(
            ["resume", "abc123", "-q", "go", "-m", "gpt-4o", "-w", "multi_agent"]
        )
        assert args.model == "gpt-4o"
        assert args.workflow == "multi_agent"


class _FakeStore:
    """Stand-in SessionStore whose get_session resolves only known ids."""

    def __init__(self, known):
        self._known = known

    def get_session(self, session_id):
        return object() if session_id in self._known else None


class TestResumeDispatch:
    def test_maps_session_id_to_resume(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            cli_main, "_handle_run", lambda args: captured.update(vars(args))
        )
        monkeypatch.setattr(
            "chemgraph.memory.store.SessionStore",
            lambda *a, **k: _FakeStore({"sess42"}),
        )
        parser = cli_main.create_argument_parser()
        args = parser.parse_args(["resume", "sess42", "-q", "continue"])
        cli_main._handle_resume(args)
        assert captured["resume"] == "sess42"
        assert captured["query"] == "continue"

    def test_prompts_when_query_omitted(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(
            cli_main, "_handle_run", lambda args: captured.update(vars(args))
        )
        monkeypatch.setattr(
            "rich.prompt.Prompt.ask", lambda *a, **k: "prompted query"
        )
        monkeypatch.setattr(
            "chemgraph.memory.store.SessionStore",
            lambda *a, **k: _FakeStore({"sess99"}),
        )
        parser = cli_main.create_argument_parser()
        args = parser.parse_args(["resume", "sess99"])
        cli_main._handle_resume(args)
        assert captured["resume"] == "sess99"
        assert captured["query"] == "prompted query"

    def test_unknown_session_id_errors_and_does_not_run(self, monkeypatch):
        import pytest

        ran = {"called": False}
        monkeypatch.setattr(
            cli_main, "_handle_run", lambda args: ran.update(called=True)
        )
        monkeypatch.setattr(
            "chemgraph.memory.store.SessionStore",
            lambda *a, **k: _FakeStore(set()),  # nothing resolves
        )
        parser = cli_main.create_argument_parser()
        args = parser.parse_args(["resume", "nope", "-q", "go"])
        with pytest.raises(SystemExit) as exc:
            cli_main._handle_resume(args)
        assert exc.value.code == 1
        assert ran["called"] is False  # never silently started a fresh run

    def test_main_routes_resume(self, monkeypatch):
        called = {}
        monkeypatch.setattr(
            cli_main, "_handle_resume", lambda args: called.setdefault("ok", args)
        )
        monkeypatch.setattr(sys, "argv", ["chemgraph", "resume", "s1", "-q", "x"])
        cli_main.main()
        assert called["ok"].session_id == "s1"
