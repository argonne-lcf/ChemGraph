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


def test_interactive_event_renders_only_tagged_subagent_tool_calls():
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
    assert "task" not in output
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
        graph_config=MainAgentGraphConfig(model_name="test-model")
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


def test_experimental_backend_can_explicitly_skip_confirmation(
    monkeypatch,
    tmp_path,
):
    class FakeBackend:
        def __init__(self, **_kwargs):
            pass

    monkeypatch.setattr("deepagents.backends.LocalShellBackend", FakeBackend)
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

    assert isinstance(backend, FakeBackend)
    assert "approvals are disabled" in capture.get().lower()


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
        model_name="argo:gpt-5.6-sol",
        structured_output=True,
        generate_report=True,
        human_supervised=True,
        recursion_limit=77,
        reasoning_effort="high",
        max_retries=4,
        terminal_tool_names=("finish",),
        topology_fingerprint="target",
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

    with console.capture():
        commands.interactive_mode(workflow="main_agent", generate_report=False)

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
            graph_config=MainAgentGraphConfig(model_name="scripted"),
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

    class Agent:
        workflow = Workflow()
        recursion_limit = 20
        return_option = "last_message"

        async def run(self, *_args, **_kwargs):
            raise HumanInputRequired("approval", payload=payload)

        def _finalize_completed_run(self, state, config, query):
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

    persisted = []

    class Agent:
        workflow = Workflow()
        recursion_limit = 20

        async def run(self, *_args, **_kwargs):
            raise HumanInputRequired("approval", payload=payloads[0])

        def _persist_run_state(self, config):
            persisted.append(config)

        def _finalize_completed_run(self, state, _config, _query):
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
                deepagent_dangerously_skip_approvals=True,
            )
        )

    assert captured["deepagent_workspace"] == str(tmp_path)
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
