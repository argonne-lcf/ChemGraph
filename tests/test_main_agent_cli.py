"""CLI tests for the long-lived main-agent workflow."""

import importlib
from types import SimpleNamespace

import pytest
import toml

from chemgraph.agent.main_session import MainAgentTurnResult, PendingInterrupt
from chemgraph.cli import commands
from chemgraph.cli.formatting import console
from chemgraph.memory.schemas import MainAgentGraphConfig, MainAgentSessionMetadata
from chemgraph.memory.store import SessionStore

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
    assert "deep_agent" in commands.ALL_WORKFLOW_TYPES
    assert commands.resolve_workflow("deepagent") == "deep_agent"
    for removed in ("python_relp", "python_repl"):
        assert removed not in commands.ALL_WORKFLOW_TYPES
        assert removed not in cli_main._WORKFLOW_CHOICES
        assert removed not in commands.WORKFLOW_ALIASES
        with pytest.raises(SystemExit):
            cli_main.create_argument_parser().parse_args(["run", "--workflow", removed])


@pytest.mark.parametrize("workflow", ["python_relp", "python_repl"])
@pytest.mark.parametrize("prefix", [[], ["run"]], ids=["legacy", "subcommand"])
def test_removed_workflow_argument_has_migration_hint(capsys, workflow, prefix):
    with pytest.raises(SystemExit) as exc_info:
        cli_main.create_argument_parser().parse_args([*prefix, "--workflow", workflow])

    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert workflow in error
    assert "has been removed" in error
    assert "deep_agent" in error
    assert "migrating-from-python-repl" in error


@pytest.mark.parametrize("workflow", ["python_relp", "python_repl"])
def test_removed_workflow_initialization_precedes_credential_check(monkeypatch, workflow):
    monkeypatch.setattr(
        commands,
        "check_api_keys",
        lambda *_args, **_kwargs: pytest.fail("removed workflow checked credentials"),
    )
    with console.capture() as capture:
        agent = commands.initialize_agent(
            model_name="gpt-4o-mini",
            workflow_type=workflow,
            structured_output=False,
            return_option="last_message",
            generate_report=False,
            recursion_limit=20,
        )

    assert agent is None
    assert "has been removed" in capture.get()
    assert "deep_agent" in capture.get()


@pytest.mark.parametrize("workflow", ["python_relp", "python_repl"])
def test_interactive_removed_workflow_keeps_current_agent(monkeypatch, workflow):
    answers = iter(["gpt-4o-mini", "single_agent", f"/workflow {workflow}", "quit"])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_args, **_kwargs: next(answers))
    initialized = []

    def initialize(model, selected_workflow, *_args, **_kwargs):
        initialized.append((model, selected_workflow))
        return SimpleNamespace(session_id="initial")

    monkeypatch.setattr(commands, "initialize_agent", initialize)
    with console.capture() as capture:
        commands.interactive_mode(workflow="single_agent", generate_report=False)

    assert initialized == [("gpt-4o-mini", "single_agent")]
    assert "has been removed" in capture.get()
    assert "deep_agent" in capture.get()


def test_interactive_event_renders_direct_and_subagent_tool_calls():
    with console.capture() as capture:
        commands._render_main_agent_event(
            "tool_call_started",
            {
                "subagent_name": "chemgraph",
                "tool_name": "run_ase",
                "arguments": "{'calculator': 'EMT'}",
            },
        )
        commands._render_main_agent_event(
            "tool_call_started",
            {"tool_name": "task", "arguments": "{'description': 'work'}"},
        )
        commands._render_main_agent_event(
            "tool_call_finished",
            {
                "subagent_name": "chemgraph",
                "tool_name": "run_ase",
                "result": "large result",
            },
        )

    output = capture.get()
    assert "chemgraph" in output
    assert "run_ase" in output
    assert "EMT" in output
    assert "main_agent" in output
    assert "task" in output
    assert "large result" not in output


def test_create_main_agent_session_installs_interactive_event_renderer(monkeypatch):
    captured = {}

    class FakeSession:
        def __init__(self, workflow, **kwargs):
            captured["workflow"] = workflow
            captured.update(kwargs)

    monkeypatch.setattr(
        "chemgraph.agent.main_session.MainAgentSession",
        FakeSession,
    )
    metadata = MainAgentSessionMetadata(
        graph_config=MainAgentGraphConfig(graph_schema_version=4, model_name="test-model")
    )
    agent = SimpleNamespace(
        workflow=object(),
        main_agent_metadata=metadata,
        session_id="thread-1",
        recursion_limit=25,
        session_store=None,
    )

    commands.create_main_agent_session(agent)

    assert captured["workflow"] is agent.workflow
    assert captured["thread_id"] == "thread-1"
    assert captured["on_event"] is commands._render_main_agent_event


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
    monkeypatch.setattr(commands.Confirm, "ask", lambda *_args, **_kwargs: True)
    monkeypatch.setenv("PATH", "/test/bin")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")

    backend = commands._create_experimental_deepagent_backend(str(tmp_path))

    from chemgraph.agent.deepagent_backend import DEEPAGENT_ENV_ALLOWLIST, create_host_shell_backend
    from chemgraph.graphs.workspace import cli_backend_descriptor

    assert backend.cwd == tmp_path.resolve()
    assert backend.virtual_mode is True
    assert backend._env["PATH"] == "/test/bin"
    assert "OPENAI_API_KEY" not in backend._env
    assert cli_backend_descriptor(backend)["type"] == "cli-local-shell-v1"
    assert backend._env == create_host_shell_backend(tmp_path)._env
    assert cli_backend_descriptor(backend)["environment_policy"] == list(DEEPAGENT_ENV_ALLOWLIST)
    assert commands._DEEPAGENT_ENV_ALLOWLIST is DEEPAGENT_ENV_ALLOWLIST


def test_experimental_backend_stops_when_confirmation_is_declined(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(commands.Confirm, "ask", lambda *_args, **_kwargs: False)

    with pytest.raises(RuntimeError, match="was not approved"):
        commands._create_experimental_deepagent_backend(str(tmp_path))


def test_experimental_backend_can_explicitly_skip_confirmation(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        commands.Confirm,
        "ask",
        lambda *_args, **_kwargs: pytest.fail("confirmation should be skipped"),
    )

    with console.capture() as capture:
        backend = commands._create_experimental_deepagent_backend(
            str(tmp_path),
            require_confirmation=False,
        )

    assert backend.cwd == tmp_path.resolve()
    rendered = " ".join(capture.get().lower().replace("│", " ").split())
    assert "approvals are disabled" in rendered


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
    assert parser.parse_args(["--workflow", "deepagent"]).workflow == "deepagent"
    assert parser.parse_args(
        ["--deepagent-dangerously-skip-approvals"]
    ).deepagent_dangerously_skip_approvals is True
    assert parser.parse_args(
        ["--deepagent-skill", "/base/", "--deepagent-skill", "/project/"]
    ).deepagent_skills == ["/base/", "/project/"]
    assert (
        parser.parse_args(["--checkpoint-db", "/tmp/checkpoints.db"]).checkpoint_db
        == "/tmp/checkpoints.db"
    )
    help_text = " ".join(parser.format_help().split())
    assert "shell commands are not confined to it" in help_text


def test_main_agent_query_failure_suggests_retry():
    session = _FakeMainSession([RuntimeError("temporary")])

    with console.capture() as capture:
        result = commands.run_main_agent_query(session, "calculate")

    assert result is None
    assert session.failed is True
    assert "`/retry` command" in capture.get()


@pytest.mark.parametrize("during_resume", [False, True])
def test_incompatible_main_agent_graph_reports_recovery_without_retry(monkeypatch, during_resume):
    from chemgraph.agent.main_session import IncompatibleCheckpointError
    from chemgraph.memory.graph_config import NEW_SESSION_GUIDANCE

    error = IncompatibleCheckpointError(f"Incompatible topology. {NEW_SESSION_GUIDANCE}")
    results = [_turn_result(PendingInterrupt("approval", {"action_requests": []}))] if during_resume else []
    session = _FakeMainSession([*results, error])
    monkeypatch.setattr(commands, "_prompt_for_interrupt", lambda _payload: {"decisions": []})
    with console.capture() as output:
        assert commands.run_main_agent_query(session, "calculate") is None
    rendered = " ".join(output.get().split())
    assert NEW_SESSION_GUIDANCE in rendered
    assert "/retry" not in rendered


def test_missing_stored_catalog_entry_reports_new_session_guidance():
    config = MainAgentGraphConfig(graph_schema_version=4, model_name="test", cli_restorable=True,
                                  topology_fingerprint="old", registry_tool_names=("removed-tool",),
                                  artifact_directory="/tmp/artifacts",
                                  model_endpoint={"endpoint_name": "test", "protocol": "test",
                                                  "requested_model": "test", "effective_model": "test"})
    with pytest.raises(ValueError, match="Start a new session; the old transcript remains readable"):
        commands._main_agent_options(config)


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


def test_workflow_switch_recovers_from_checkpoint_open_failure(monkeypatch):
    class FakeRuntime:
        def __init__(self, *, error=None):
            self.error = error
            self.closed = False
            self.saver = SimpleNamespace()
            self.opened_paths = []

        def open_sqlite(self, path):
            self.opened_paths.append(path)
            if self.error is not None:
                raise self.error
            return self.saver

        def close(self):
            self.closed = True

    failed_runtime = FakeRuntime(error=RuntimeError("database is locked"))
    successful_runtime = FakeRuntime()
    runtimes = iter([failed_runtime, successful_runtime])
    initial_agent = SimpleNamespace(session_id="initial")
    main_agent = SimpleNamespace()
    agents = iter([initial_agent, main_agent])
    initialization_calls = []
    answers = iter(
        [
            "first-model",
            "single_agent",
            "/workflow main_agent",
            "/workflow main_agent",
            "quit",
        ]
    )

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "CheckpointRuntime", lambda: next(runtimes))

    def fake_initialize(_model, workflow, *_args, **kwargs):
        initialization_calls.append((workflow, kwargs["checkpointer"]))
        return next(agents)

    monkeypatch.setattr(commands, "initialize_agent", fake_initialize)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda *_args, **_kwargs: SimpleNamespace(thread_id="main", failed=False),
    )

    with console.capture() as capture:
        commands.interactive_mode(workflow="single_agent", generate_report=False)

    assert failed_runtime.closed is True
    assert successful_runtime.closed is True
    assert initialization_calls == [
        ("single_agent", None),
        ("main_agent", successful_runtime.saver),
    ]
    output = capture.get()
    assert "Could not open checkpoint database: database is locked" in output
    assert "Workflow changed to: main_agent" in output


def test_resume_replaces_all_active_graph_settings(monkeypatch, tmp_path):
    target_config = MainAgentGraphConfig(
        graph_schema_version=4,
        artifact_directory=str(tmp_path / "artifacts"),
        model_endpoint={"endpoint_name": "argo", "protocol": "openai_compatible",
                        "requested_model": "argo:gpt-5.6-sol", "effective_model": "gpt-5.6-sol"},
        cli_restorable=True,
        model_name="argo:gpt-5.6-sol",
        structured_output=True,
        generate_report=True,
        human_supervised=True,
        recursion_limit=77,
        reasoning_effort="high",
        max_retries=4,
        terminal_tool_names=("finish",),
        enable_deepagent=True,
        deepagent_workspace=str(tmp_path),
        deepagent_skills=("/workspace/.agents/skills/",),
        deepagent_skill_dirs=(str(tmp_path.resolve()),),
        deepagent_discover_skills=True,
        deepagent_user_skills_dir=str(tmp_path / "personal-skills"),
        topology_fingerprint="target",
        workspace=str(tmp_path),
        skills=("/workspace/new-skills/",),
        skill_dirs=(str(tmp_path),),
        discover_skills=False,
        registry_tool_names=("calculator",),
        configured_subagent_names=("single_agent",),
        subagent_names=("single_agent", "deep_agent"),
        main_agent_prompt="stored main prompt",
    )
    target_db = str(tmp_path / "target-checkpoints.db")
    SessionStore().create_session(
        "target-thread",
        target_config.model_name,
        "main_agent",
        session_metadata=MainAgentSessionMetadata(
            graph_config=target_config,
            checkpoint_backend="AsyncSqliteSaver",
            checkpoint_db=target_db,
        ),
    )
    answers = iter(
        [
            "initial-model",
            "main_agent",
            "/resume target-thread",
            "/workflow single_agent",
            "quit",
        ]
    )
    agents = iter([SimpleNamespace(), SimpleNamespace(), SimpleNamespace(session_id="new")])
    initialization_calls = []
    sessions = iter(
        [
            SimpleNamespace(thread_id="initial", failed=False),
            SimpleNamespace(thread_id="target-thread", failed=False),
        ]
    )

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )

    def fake_initialize(*args, **kwargs):
        initialization_calls.append((args, kwargs))
        return next(agents)

    monkeypatch.setattr(commands, "initialize_agent", fake_initialize)
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda *_args, **_kwargs: next(sessions),
    )
    monkeypatch.setattr(
        commands,
        "restore_main_agent_session",
        lambda *_args, **_kwargs: MainAgentTurnResult(
            thread_id="target-thread",
            status="completed",
            assistant_response="",
            interrupts=(),
            state={},
        ),
    )

    with console.capture() as output:
        commands.interactive_mode(workflow="main_agent", generate_report=False)

    rendered = " ".join(output.get().split())
    assert "Using saved main-agent configuration for session target-thread." in rendered
    assert "Saved graph settings take precedence over current CLI flags and TOML settings." in rendered
    assert "calculator" in rendered and "single_agent" in rendered
    assert "/workspace/new-skills/" in rendered
    resume_args, resume_kwargs = initialization_calls[1]
    rebuild_args, rebuild_kwargs = initialization_calls[2]
    assert resume_args[:6] == (
        target_config.model_name,
        "main_agent",
        True,
        "state",
        True,
        77,
    )
    assert rebuild_args[:6] == (
        target_config.model_name,
        "single_agent",
        True,
        "state",
        True,
        77,
    )
    assert resume_kwargs["workspace"] == str(tmp_path)
    assert resume_kwargs["skills"] == ("/workspace/new-skills/",)
    assert resume_kwargs["skill_dirs"] == (str(tmp_path),)
    assert resume_kwargs["discover_skills"] is False
    assert resume_kwargs["tool_registry"].names() == ("calculator",)
    assert resume_kwargs["subagent_names"] == ("single_agent",)
    assert resume_kwargs["main_agent_prompt"] == "stored main prompt"
    assert "workspace" not in rebuild_kwargs
    assert resume_kwargs["deepagent_skills"] == (
        "/workspace/.agents/skills/",
    )
    assert rebuild_kwargs["deepagent_skills"] is None
    assert resume_kwargs["deepagent_skill_dirs"] == (str(tmp_path.resolve()),)
    assert rebuild_kwargs["deepagent_skill_dirs"] is None
    for kwargs in (resume_kwargs, rebuild_kwargs):
        assert kwargs["deepagent_discover_skills"] is True
        assert kwargs["deepagent_user_skills_dir"] == str(tmp_path / "personal-skills")
    for kwargs in (resume_kwargs, rebuild_kwargs):
        assert kwargs["human_supervised"] is True
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["max_retries"] == 4
        assert kwargs["terminal_tool_names"] == ("finish",)


def test_startup_resume_distinguishes_process_local_session(monkeypatch):
    SessionStore().create_session(
        "process-local",
        "scripted",
        "main_agent",
        session_metadata=MainAgentSessionMetadata(
            graph_config=MainAgentGraphConfig(graph_schema_version=4, model_name="scripted"),
            checkpoint_backend="memory",
        ),
    )
    monkeypatch.setattr(
        commands,
        "initialize_agent",
        lambda *_args, **_kwargs: pytest.fail("process-local session was initialized"),
    )

    with console.capture() as capture:
        commands.interactive_mode(resume_session="process-local")

    assert "process-local checkpoint" in capture.get()


def test_interactive_eof_closes_checkpoint_runtime(monkeypatch):
    class FakeRuntime:
        def __init__(self):
            self.closed = False

        def open_sqlite(self, _path):
            return SimpleNamespace()

        def close(self):
            self.closed = True

    runtime = FakeRuntime()
    answers = iter(["model", "main_agent"])

    def prompt(*_args, **_kwargs):
        try:
            return next(answers)
        except StopIteration as exc:
            raise EOFError from exc

    monkeypatch.setattr(commands, "CheckpointRuntime", lambda: runtime)
    monkeypatch.setattr(commands.Prompt, "ask", prompt)
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        commands,
        "create_main_agent_session",
        lambda *_args, **_kwargs: SimpleNamespace(thread_id="main", failed=False),
    )

    with console.capture():
        commands.interactive_mode(workflow="main_agent")

    assert runtime.closed is True


def test_interactive_deepagent_setting_survives_workflow_switches(monkeypatch, tmp_path):
    answers = iter(
        [
            "first-model",
            "main_agent",
            "/model second-model",
            "/workflow single_agent",
            "/workflow main_agent",
            "quit",
        ]
    )
    agents = iter(
        [
            SimpleNamespace(),
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
            (
                kwargs["enable_deepagent"],
                kwargs["deepagent_workspace"],
                kwargs["deepagent_skills"],
                kwargs["deepagent_skill_dirs"],
            )
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
            deepagent_skills=["/workspace/.agents/skills/"],
            deepagent_skill_dirs=[str(tmp_path)],
        )

    assert initialization_calls == [
        (True, "/workspace", ["/workspace/.agents/skills/"], (str(tmp_path),)),
        (True, "/workspace", ["/workspace/.agents/skills/"], (str(tmp_path),)),
        (False, None, None, None),
        (True, "/workspace", ["/workspace/.agents/skills/"], (str(tmp_path),)),
    ]


@pytest.fixture
def skill_repl(monkeypatch):
    created, shells, queries = [], [], []

    def create(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(session_id="test", **kwargs)

    monkeypatch.setattr("chemgraph.agent.llm_agent.ChemGraph", create)
    monkeypatch.setattr(commands, "check_api_keys", lambda *_, **__: (True, None))
    monkeypatch.setattr(commands.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        commands, "_create_experimental_deepagent_backend",
        lambda *args, **kwargs: shells.append((args, kwargs)) or object(),
    )
    monkeypatch.setattr(
        commands, "create_main_agent_session",
        lambda *_, **__: SimpleNamespace(thread_id="main", failed=False),
    )
    monkeypatch.setattr(
        commands, "run_query",
        lambda agent, *_, **__: queries.append(agent.workflow_type),
    )
    return created, shells, queries


@pytest.mark.parametrize("startup_workflow", ["single_agent", "deep_agent"])
@pytest.mark.parametrize("target_workflow", ["deep_agent", "main_agent"])
@pytest.mark.parametrize("via_cli", [False, True])
def test_interactive_defers_skill_access_and_retries_switch(
    monkeypatch, tmp_path, skill_repl, startup_workflow, target_workflow, via_cli
):
    collection = tmp_path / "external skills"
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    monkeypatch.chdir(tmp_path)

    def answers():
        yield "first-model"
        monkeypatch.chdir(other_cwd)
        yield "single_agent"
        yield "/model second-model"
        yield f"/workflow {target_workflow}"
        yield "still using the current agent"
        collection.mkdir()
        yield f"/workflow {target_workflow}"
        yield "/model third-model"
        yield "quit"

    responses = answers()
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_, **__: next(responses))
    settings = dict(
        workflow=startup_workflow,
        enable_deepagent=target_workflow == "main_agent",
        deepagent_workspace=str(tmp_path),
    )
    with console.capture() as capture:
        if via_cli:
            path = tmp_path / "config.toml"
            path.write_text(toml.dumps({"general": {
                **settings, "deepagent_skills": ["./external skills"],
            }}))
            cli_main._handle_run(cli_main.create_argument_parser().parse_args([
                "run", "--interactive", "--config", str(path),
            ]))
        else:
            commands.interactive_mode(
                **settings, deepagent_skill_dirs=["./external skills"],
            )

    created, shells, queries = skill_repl
    assert [item["workflow_type"] for item in created] == [
        "single_agent", "single_agent", target_workflow, target_workflow,
    ]
    assert [item["deepagent_skill_dirs"] for item in created] == [
        (), (), (str(collection),), (str(collection),),
    ]
    assert len(shells) == 2  # Failed validation never enables the shell.
    assert queries == ["single_agent"]
    assert "Cannot access skill directory" in capture.get()
    assert capture.get().count(f"Workflow changed to: {target_workflow}") == 1


@pytest.mark.parametrize("activate_at_startup", [False, True])
def test_interactive_retains_canonical_skills_after_activation(
    monkeypatch, tmp_path, skill_repl, activate_at_startup
):
    original, replacement = tmp_path / "original", tmp_path / "replacement"
    original.mkdir()
    replacement.mkdir()
    link = tmp_path / "skills"
    link.symlink_to(original, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    def answers():
        yield "first-model"
        yield "deep_agent" if activate_at_startup else "single_agent"
        if not activate_at_startup:
            yield "/workflow deep_agent"
        link.unlink()
        link.symlink_to(replacement, target_is_directory=True)
        yield "/model second-model"
        yield "/workflow single_agent"
        yield "/workflow deep_agent"
        yield "quit"

    responses = answers()
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_, **__: next(responses))
    with console.capture():
        commands.interactive_mode(deepagent_skill_dirs=["./skills"])

    created, _, _ = skill_repl
    active_dirs = [
        item["deepagent_skill_dirs"] for item in created
        if item["workflow_type"] == "deep_agent"
    ]
    assert active_dirs == [(str(original),)] * 3


def test_interactive_invalid_active_directory_fails_before_shell(
    monkeypatch, tmp_path, skill_repl
):
    responses = iter(["fake-model", "deep_agent"])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *_, **__: next(responses))
    with console.capture() as capture:
        commands.interactive_mode(deepagent_skill_dirs=[str(tmp_path / "missing")])
    created, shells, _ = skill_repl
    assert not created and not shells
    assert "Cannot access skill directory" in capture.get()


def test_interactive_standalone_deepagent_reuses_one_thread(monkeypatch):
    answers = iter(
        ["gpt-4o-mini", "deep_agent", "inspect files", "run tests", "quit"]
    )
    agent = SimpleNamespace(session_id="deep-session")
    thread_ids = []

    monkeypatch.setattr(
        commands.Prompt,
        "ask",
        lambda *_args, **_kwargs: next(answers),
    )
    monkeypatch.setattr(commands, "initialize_agent", lambda *_args, **_kwargs: agent)

    def fake_run(_agent, _query, *, thread_id, verbose=False):
        thread_ids.append(thread_id)
        return None

    monkeypatch.setattr(commands, "run_query", fake_run)

    with console.capture():
        commands.interactive_mode(
            workflow="deep_agent",
            generate_report=False,
            deepagent_workspace="/workspace",
        )

    assert len(thread_ids) == 2
    assert thread_ids[0] == thread_ids[1]


def test_run_query_preserves_structured_deepagent_approval(monkeypatch):
    from chemgraph.agent.llm_agent import HumanInputRequired

    payload = {
        "action_requests": [{"name": "execute", "args": {"command": "ruff"}}],
        "review_configs": [
            {
                "action_name": "execute",
                "allowed_decisions": ["approve", "reject"],
            }
        ],
    }
    resume_inputs = []
    finalized = []

    class Workflow:
        async def astream(self, stream_input, **_kwargs):
            resume_inputs.append(stream_input)
            yield {"messages": ["done"]}

        def get_state(self, _config):
            return SimpleNamespace(tasks=())

        async def aget_state(self, config):
            return self.get_state(config)

    class Agent:
        workflow = Workflow()
        recursion_limit = 20
        return_option = "last_message"

        async def run(self, *_args, **_kwargs):
            raise HumanInputRequired("approval", payload=payload)

        async def afinalize_completed_run(self, state, config, query):
            finalized.append((state, config, query))
            return state["messages"][-1]

    prompted = []
    monkeypatch.setattr(
        commands,
        "_prompt_for_interrupt",
        lambda value: prompted.append(value)
        or {"decisions": [{"type": "approve"}]},
    )

    result = commands.run_query(Agent(), "run lint", thread_id=10)

    assert result == "done"
    assert prompted == [payload]
    assert resume_inputs[0].resume == {"decisions": [{"type": "approve"}]}
    assert finalized == [
        (
            {"messages": ["done"]},
            {
                "configurable": {"thread_id": 10},
                "recursion_limit": 20,
            },
            "run lint",
        )
    ]


def test_run_query_maps_multiple_interrupt_responses_by_id(monkeypatch):
    from chemgraph.agent.llm_agent import HumanInputRequired

    payloads = [
        {"question": "First approval?"},
        {"question": "Second approval?"},
    ]
    pending = (
        PendingInterrupt(id="first-id", payload=payloads[0]),
        PendingInterrupt(id="second-id", payload=payloads[1]),
    )
    resume_inputs = []

    class Workflow:
        async def astream(self, stream_input, **_kwargs):
            resume_inputs.append(stream_input)
            yield {"messages": ["done"]}

        def get_state(self, _config):
            return SimpleNamespace(tasks=(), interrupts=())

        async def aget_state(self, config):
            return self.get_state(config)

    class Agent:
        workflow = Workflow()
        recursion_limit = 20

        async def run(self, *_args, **_kwargs):
            raise HumanInputRequired(
                "First approval?",
                payload=payloads[0],
                interrupts=pending,
            )

        async def afinalize_completed_run(self, state, _config, _query):
            return state["messages"][-1]

    answers = iter(["approve-first", "reject-second"])
    prompted = []
    monkeypatch.setattr(
        commands,
        "_prompt_for_interrupt",
        lambda payload: prompted.append(payload) or next(answers),
    )

    result = commands.run_query(Agent(), "review actions", thread_id=12)

    assert result == "done"
    assert prompted == payloads
    assert resume_inputs[0].resume == {
        "first-id": "approve-first",
        "second-id": "reject-second",
    }


def test_run_query_rejects_multiple_interrupts_without_stable_ids(monkeypatch):
    from chemgraph.agent.llm_agent import HumanInputRequired

    pending = (
        PendingInterrupt(id="", payload={"question": "First?"}),
        PendingInterrupt(id="second-id", payload={"question": "Second?"}),
    )

    class Agent:
        recursion_limit = 20

        async def run(self, *_args, **_kwargs):
            raise HumanInputRequired(
                "First?",
                payload=pending[0].payload,
                interrupts=pending,
            )

    monkeypatch.setattr(commands, "_prompt_for_interrupt", lambda payload: payload)

    with console.capture() as capture:
        result = commands.run_query(Agent(), "review actions", thread_id=13)

    assert result is None
    assert "do not expose stable IDs" in capture.get()


def test_run_query_persists_chained_deepagent_interrupt(monkeypatch):
    from chemgraph.agent.llm_agent import HumanInputRequired

    payloads = [
        {"action_requests": [{"name": "write_file", "args": {}}]},
        {"action_requests": [{"name": "execute", "args": {}}]},
    ]

    class Workflow:
        resume_count = 0

        async def astream(self, _stream_input, **_kwargs):
            self.resume_count += 1
            if self.resume_count == 1:
                yield {
                    "messages": ["waiting"],
                    "__interrupt__": [SimpleNamespace(value=payloads[1])],
                }
            else:
                yield {"messages": ["done"]}

        def get_state(self, _config):
            return SimpleNamespace(tasks=())

        async def aget_state(self, config):
            return self.get_state(config)

    persisted = []

    class Agent:
        workflow = Workflow()
        recursion_limit = 20

        async def run(self, *_args, **_kwargs):
            raise HumanInputRequired("approval", payload=payloads[0])

        async def apersist_run_state(self, config):
            persisted.append(config)

        async def afinalize_completed_run(self, state, _config, _query):
            return state["messages"][-1]

    prompted = []
    monkeypatch.setattr(
        commands,
        "_prompt_for_interrupt",
        lambda payload: prompted.append(payload)
        or {"decisions": [{"type": "approve"}]},
    )

    result = commands.run_query(Agent(), "write and run", thread_id=11)

    assert result == "done"
    assert prompted == payloads
    assert persisted == [
        {
            "configurable": {"thread_id": 11},
            "recursion_limit": 20,
        }
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
        "deepagent_skills": None,
        "deepagent_dangerously_skip_approvals": False,
        "query": None,
        "output_file": None,
        "trace_dir": None,
        "checkpoint_db": None,
        "mcp_url": None,
        "mcp_command": None,
        "mcp_server_name": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("workflow", ["python_relp", "python_repl"])
def test_removed_workflow_in_toml_has_migration_hint(monkeypatch, tmp_path, workflow):
    path = tmp_path / "legacy.toml"
    path.write_text(toml.dumps({"general": {"workflow": workflow}}))
    monkeypatch.setattr(
        cli_main,
        "initialize_agent",
        lambda *_args, **_kwargs: pytest.fail("removed config reached agent initialization"),
    )
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        cli_main._handle_run(_run_args(config=str(path), workflow=None))

    assert exc_info.value.code == 2
    assert workflow in capture.get()
    assert "has been removed" in capture.get()
    assert "deep_agent" in capture.get()


def test_main_agent_requires_interactive_cli_mode():
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        cli_main._handle_run(_run_args())

    assert exc_info.value.code == 2
    assert "requires interactive mode" in capture.get()


def test_headless_deepagent_requires_unsafe_flag_and_workspace(tmp_path):
    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        cli_main._handle_run(
            _run_args(
                workflow="deep_agent",
                query="inspect the repository",
                deepagent_workspace=str(tmp_path),
            )
        )

    assert exc_info.value.code == 2
    assert "dangerously-skip-approvals" in capture.get()

    with console.capture() as capture, pytest.raises(SystemExit) as exc_info:
        cli_main._handle_run(
            _run_args(
                workflow="deep_agent",
                query="inspect the repository",
                deepagent_dangerously_skip_approvals=True,
            )
        )

    assert exc_info.value.code == 2
    assert "explicit --deepagent-workspace" in capture.get()


def test_headless_deepagent_forwards_explicit_unsafe_configuration(
    monkeypatch,
    tmp_path,
):
    captured = {}
    agent = SimpleNamespace(session_id="deep-session")
    monkeypatch.setattr(
        cli_main,
        "initialize_agent",
        lambda *_args, **kwargs: captured.update(kwargs) or agent,
    )
    monkeypatch.setattr(cli_main, "run_query", lambda *_args, **_kwargs: None)

    with console.capture():
        cli_main._handle_run(
            _run_args(
                workflow="deep_agent",
                query="inspect the repository",
                deepagent_workspace=str(tmp_path),
                deepagent_skills=[str(tmp_path)],
                deepagent_dangerously_skip_approvals=True,
            )
        )

    assert captured["deepagent_workspace"] == str(tmp_path)
    assert captured["deepagent_skill_dirs"] == (str(tmp_path.resolve()),)
    assert captured["deepagent_auto_approve"] is True


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
                    "deepagent_skills": [str(tmp_path)],
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
    assert captured["deepagent_skill_dirs"] == (
        (str(tmp_path.resolve()),) if expected else None
    )


@pytest.mark.parametrize("override", [False, True])
def test_main_agent_cli_and_toml_options(monkeypatch, tmp_path, override):
    config = tmp_path / "config.toml"
    config.write_text(toml.dumps({"general": {
        "workflow": "main_agent", "workspace": str(tmp_path),
        "skills": [str(tmp_path)], "discover_skills": True,
        "subagents": ["single_agent"], "tools": ["calculator"],
    }}))
    parser = cli_main.create_argument_parser()
    argv = ["run", "--interactive", "--config", str(config)]
    if override:
        argv += ["--workspace", str(tmp_path / "other"), "--skill", str(tmp_path / "extra"),
                 "--no-discover-skills", "--subagent", "deep_agent", "--tool", "run_ase"]
    captured = {}
    monkeypatch.setattr(cli_main, "interactive_mode", lambda **kwargs: captured.update(kwargs))
    cli_main._handle_run(parser.parse_args(argv))
    assert captured["workspace"] == str(tmp_path / "other" if override else tmp_path)
    assert captured["skill_dirs"] == [str(tmp_path / "extra" if override else tmp_path)]
    assert captured["discover_skills"] is (not override)
    assert captured["subagent_names"] == (["deep_agent"] if override else ["single_agent"])
    assert captured["tool_registry"].names() == (("run_ase",) if override else ("calculator",))
    assert captured["deepagent_tool_registry"] is captured["tool_registry"]
    assert captured["enable_deepagent"] is False


def test_main_workspace_options_survive_workflow_and_model_switches(monkeypatch, tmp_path):
    from chemgraph.registry.tools import ToolRegistry

    replies = iter(["test-model", "single_agent", "/workflow main_agent",
                    "/model other-model", "/workflow single_agent", "/workflow main_agent", "/quit"])
    calls = []
    monkeypatch.setattr(commands.Prompt, "ask", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(commands, "initialize_agent",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace())
    monkeypatch.setattr(commands, "create_main_agent_session",
                        lambda *args, **kwargs: SimpleNamespace(thread_id="test"))
    registry = ToolRegistry([])
    with console.capture():
        commands.interactive_mode(workspace=str(tmp_path), skill_dirs=[str(tmp_path)],
                                  discover_skills=False, tool_registry=registry,
                                  subagent_names=["single_agent"])
    main_calls = [kwargs for args, kwargs in calls if args[1] == "main_agent"]
    assert len(main_calls) == 3
    for kwargs in main_calls:
        assert kwargs["workspace"] == str(tmp_path)
        assert kwargs["skill_dirs"] == (str(tmp_path.resolve()),)
        assert kwargs["discover_skills"] is False
        assert kwargs["tool_registry"] is registry
        assert kwargs["subagent_names"] == ["single_agent"]
    for args, kwargs in calls:
        if args[1] != "main_agent":
            assert "workspace" not in kwargs


def test_legacy_cli_resume_fails_before_backend_initialization(monkeypatch, tmp_path):
    SessionStore().create_session("legacy", "old", "main_agent", session_metadata=MainAgentSessionMetadata(
        graph_config=MainAgentGraphConfig(model_name="old"),
        checkpoint_backend="AsyncSqliteSaver", checkpoint_db=str(tmp_path / "old.db"),
    ))
    monkeypatch.setattr(commands, "initialize_agent",
                        lambda *args, **kwargs: pytest.fail("legacy graph must not initialize"))
    with console.capture() as output:
        commands.interactive_mode(resume_session="legacy")
    assert "Start a new session" in output.get()
    assert SessionStore().get_session("legacy") is not None


@pytest.mark.parametrize("starting_workflow", ["main_agent", "deep_agent"])
def test_catalog_survives_full_cli_dispatch_and_workflow_switch(monkeypatch, starting_workflow):
    replies = iter(["test-model", "single_agent", "/workflow main_agent",
                    "/workflow deep_agent", "/workflow main_agent", "/quit"])
    calls = []
    monkeypatch.setattr(commands.Prompt, "ask", lambda *args, **kwargs: next(replies))
    monkeypatch.setattr(commands, "initialize_agent",
                        lambda *args, **kwargs: calls.append((args[1], kwargs)) or SimpleNamespace())
    monkeypatch.setattr(commands, "create_main_agent_session",
                        lambda *args, **kwargs: SimpleNamespace(thread_id="test"))
    with console.capture():
        cli_main._handle_run(cli_main.create_argument_parser().parse_args([
            "run", "--interactive", "-w", starting_workflow, "--tool", "calculator",
        ]))
    for workflow, options in calls:
        if workflow in {"main_agent", "deep_agent"}:
            registry = options["tool_registry" if workflow == "main_agent" else "deepagent_tool_registry"]
            assert registry.names() == ("calculator",)


@pytest.mark.parametrize("workspace", ["", 123, False])
def test_invalid_main_workspace_fails_before_backend_creation(monkeypatch, workspace):
    monkeypatch.setattr(commands, "_create_experimental_deepagent_backend",
                        lambda *args, **kwargs: pytest.fail("must validate first"))
    with console.capture() as output:
        assert commands.initialize_agent("test", "main_agent", False, "state", False, 200,
                                         workspace=workspace) is None
    assert "non-empty host directory" in output.get()


@pytest.mark.parametrize("startup", [False, True])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_cli_reconstructs_pending_workspace_action_from_sqlite(monkeypatch, tmp_path, startup, decision):
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.agent.main_session import MainAgentSession
    from chemgraph.cli.checkpoint_runtime import CheckpointRuntime
    from chemgraph.graphs.workspace import _CLIWorkspaceBackend, create_cli_workspace_backend
    from chemgraph.models.endpoints import PreparedModel
    from chemgraph.registry.tools import ToolRegistry
    from langchain_core.messages import AIMessage
    from tests.test_main_agent import _ScriptedChatModel

    saved_workspace, supplied_workspace = tmp_path / "saved", tmp_path / "supplied"
    saved_workspace.mkdir()
    supplied_workspace.mkdir()
    executions = []
    write = _CLIWorkspaceBackend.write
    def record_write(self, *args, **kwargs):
        executions.append(str(self.cwd))
        return write(self, *args, **kwargs)
    monkeypatch.setattr(_CLIWorkspaceBackend, "write", record_write)
    responses = [AIMessage(content="", tool_calls=[{
        "name": "write_file", "args": {"file_path": "/workspace/review.txt", "content": "approved"},
        "id": "write-1", "type": "tool_call",
    }])]
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", lambda **kwargs: (
        _ScriptedChatModel(responses=list(responses)),
        PreparedModel(endpoint_name="test", protocol="openai_compatible", client_kwargs={}),
    ))
    monkeypatch.setattr(commands, "check_api_keys", lambda *args, **kwargs: (True, ""))
    monkeypatch.setattr(commands.Confirm, "ask", lambda *args, **kwargs: True)
    monkeypatch.setattr(commands.time, "sleep", lambda *_args: None)
    database = str(tmp_path / "checkpoints.db")
    runtime = CheckpointRuntime()
    try:
        agent = ChemGraph(workflow_type="main_agent", model_name="test", enable_memory=False,
                          backend=create_cli_workspace_backend(saved_workspace), discover_skills=False,
                          tool_registry=ToolRegistry([]), log_dir=str(tmp_path / "logs"),
                          checkpointer=runtime.open_sqlite(database))
        metadata = agent.main_agent_metadata.model_copy(deep=True)
        metadata.checkpoint_db = database
        session = MainAgentSession(agent.workflow, thread_id="pending-cli", session_metadata=metadata,
                                   session_store=SessionStore())
        assert runtime.run(lambda: session.run("write the file")).status == "waiting_for_user"
    finally:
        runtime.close()
    responses[:] = [AIMessage(content=f"Decision: {decision}.")]
    replies = iter([decision, "/quit"] if startup else
                   ["test", "main_agent", "/resume pending-cli", decision, "/quit"])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *args, **kwargs: next(replies))
    with console.capture() as output:
        commands.interactive_mode(model="test", workflow="main_agent", checkpoint_db=database,
                                  workspace=str(supplied_workspace), skill_dirs=[str(supplied_workspace)],
                                  subagent_names=["single_agent"],
                                  tool_registry=ToolRegistry([ToolRegistry().get_spec("calculator")]),
                                  resume_session="pending-cli" if startup else None)
    rendered = " ".join(output.get().split())
    assert f"Decision: {decision}." in rendered
    assert "Using saved main-agent configuration for session pending-cli." in rendered
    assert "Saved graph settings take precedence over current CLI flags and TOML settings." in rendered
    compact = "".join(rendered.split())
    assert str(saved_workspace) in compact
    assert rendered.index("Using saved main-agent") < rendered.rindex("Experimental host-shell access")
    assert executions == ([str(saved_workspace)] if decision == "approve" else [])
    assert not (supplied_workspace / "review.txt").exists()
    if decision == "approve":
        assert (saved_workspace / "review.txt").read_text() == "approved"
    else:
        assert not (saved_workspace / "review.txt").exists()
    assert SessionStore().get_session("pending-cli").status == "completed"


def test_main_workspace_symlink_is_frozen_after_activation(monkeypatch, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    workspace = tmp_path / "workspace"
    workspace.symlink_to(first, target_is_directory=True)
    calls = []
    replies = iter(["test", "main_agent", "/model changed", "/quit"])
    def initialize(*args, **kwargs):
        calls.append(kwargs["workspace"])
        return SimpleNamespace(backend=SimpleNamespace(cwd=first.resolve()))
    def ask(*args, **kwargs):
        reply = next(replies)
        if reply == "/model changed":
            workspace.unlink()
            workspace.symlink_to(second, target_is_directory=True)
        return reply
    monkeypatch.setattr(commands, "initialize_agent", initialize)
    monkeypatch.setattr(commands.Prompt, "ask", ask)
    monkeypatch.setattr(commands, "create_main_agent_session",
                        lambda *args, **kwargs: SimpleNamespace(thread_id="test"))
    with console.capture():
        commands.interactive_mode(workflow="main_agent", workspace=str(workspace))
    assert calls == [str(workspace), str(first.resolve())]
