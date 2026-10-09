"""Approval policy, unattended execution, and durable launch authorization."""

from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from chemgraph.agent.approvals import normalize_approval_mode
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.agent.main_session import MainAgentSession
from chemgraph.cli import commands
from chemgraph.cli.headless import run_headless_main_agent
from chemgraph.memory.store import SessionStore
from chemgraph.registry.tools import ToolRegistry
from chemgraph.graphs.workspace import create_cli_workspace_backend
from tests.test_main_agent_configuration import api as api
from tests.test_main_agent_execution_context import model_loader as model_loader, call
from tests.test_main_agent_cli import cli_main, _run_args


@pytest.fixture(autouse=True)
def artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "artifacts"))


@pytest.mark.parametrize("workflow,mode,legacy", [
    ("main_agent", "invalid", False), ("single_agent", "bypass", False),
    ("main_agent", None, True), ("deep_agent", "review", True),
])
def test_invalid_modes_fail_before_loading_model(monkeypatch, workflow, mode, legacy):
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared",
                        lambda **kwargs: pytest.fail("model must not be loaded"))
    with pytest.raises(ValueError):
        ChemGraph(workflow_type=workflow, approval_mode=mode, deepagent_auto_approve=legacy)


def test_mode_round_trip_and_reviewed_identity(api, monkeypatch):
    from chemgraph.agent import configuration
    from chemgraph.memory.graph_config import fingerprint

    create, captured = api
    identities = []
    monkeypatch.setattr(configuration, "fingerprint", lambda payload: identities.append(payload) or fingerprint(payload))
    reviewed = create().runtime_config.saved
    bypass = create(approval_mode="bypass").runtime_config.saved
    assert captured["interrupt_on"] is None
    assert configuration.restoration_arguments(bypass)["approval_mode"] == "bypass"
    assert reviewed.topology_fingerprint != bypass.topology_fingerprint
    # Old metadata and old fingerprint inputs must reconstruct the reviewed graph.
    legacy = reviewed.model_dump(exclude={"approval_mode"})
    restored = type(reviewed).model_validate(legacy)
    assert restored.approval_mode == "review"
    assert "approval_mode" not in identities[0]
    assert create(approval_mode=restored.approval_mode).runtime_config.saved.topology_fingerprint == reviewed.topology_fingerprint
    assert normalize_approval_mode("deep_agent", deepagent_auto_approve=True) == "bypass"


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["direct", "catalog", "worker", "strict_worker"])
@pytest.mark.parametrize("mode", ["review", "bypass"])
async def test_native_calculations_follow_shared_policy(model_loader, route, mode):
    responses, _ = model_loader
    executions = []

    @tool
    def run_ase() -> str:
        """Hermetic calculation stub."""
        executions.append("calculated")
        return "energy=1 eV"

    options = {"subagent_names": [], "tool_registry": ToolRegistry([])}
    if route == "direct":
        options["tools"] = [run_ase]
    elif route == "catalog":
        options["tool_registry"].register(run_ase)
        responses.append(call("load_tools", names=["run_ase"]))
    else:
        options.update(subagent_names=["single_agent"], subagent_options={"single_agent": {"tools": [run_ase]}})
        if route == "strict_worker":
            options["subagent_options"]["single_agent"]["interrupt_on"] = {"run_ase": True}
        responses.extend([call("load_agents", names=["single_agent"]),
                          call("task", subagent_type="single_agent", description="calculate")])
    responses.extend([call("run_ase"), AIMessage(content="done"), AIMessage(content="main done")])
    agent = ChemGraph(workflow_type="main_agent", approval_mode=mode, discover_skills=False, **options)
    result = await MainAgentSession(agent.workflow, session_metadata=agent.main_agent_metadata).run("calculate")
    executed = mode == "bypass" and route != "strict_worker"
    assert result.status == ("completed" if executed else "waiting_for_user")
    assert executions == (["calculated"] if executed else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["main_agent", "worker", "legacy", "deep_agent"])
async def test_workspace_bypass_reaches_shell_and_file_tools(tmp_path, model_loader, route):
    responses, _ = model_loader
    backend = create_cli_workspace_backend(tmp_path)
    options = {"backend": backend, "subagent_names": [], "tool_registry": ToolRegistry([])}
    options["discover_skills"] = False
    if route in {"worker", "legacy"}:
        if route == "worker":
            options["subagent_names"] = ["deep_agent"]
            responses.append(call("load_agents", names=["deep_agent"]))
        else:
            options.update(subagent_names=None, enable_deepagent=True, deepagent_backend=backend,
                           deepagent_discover_skills=False)
        responses.append(call("task", subagent_type="deep_agent", description="write the file"))
    elif route == "deep_agent":
        options = {"deepagent_backend": backend, "deepagent_discover_skills": False,
                   "deepagent_tool_registry": ToolRegistry([])}
    responses.extend([call("write_file", file_path="/workspace/result.txt", content="autonomous"),
                      call("execute", command="printf autonomous"), AIMessage(content="done"), AIMessage(content="main done")])
    agent = ChemGraph(workflow_type="deep_agent" if route == "deep_agent" else "main_agent",
                      approval_mode="bypass", **options)
    result = await MainAgentSession(agent.workflow).run("write")
    assert result.status == "completed"
    assert (tmp_path / "result.txt").read_text() == "autonomous"


@pytest.fixture
def headless(monkeypatch, tmp_path, model_loader):
    def forbidden(*args, **kwargs):
        pytest.fail("headless execution must not prompt")
    monkeypatch.setattr(commands.Prompt, "ask", forbidden)
    monkeypatch.setattr(commands.Confirm, "ask", forbidden)
    options = dict(model_name="test", structured_output=False, return_option="state", generate_report=False,
                   recursion_limit=200, approval_mode="bypass", discover_skills=False,
                   subagent_names=[], tool_registry=ToolRegistry([]),
                   checkpoint_db=str(tmp_path / "checkpoints.db"))
    def run(**changes):
        return run_headless_main_agent(**{**options, **changes})
    return run, model_loader[0], model_loader[1]


def test_headless_restart_does_not_replay_completed_tools(headless, tmp_path):
    run, responses, requests = headless
    responses[:] = [call("write_file", file_path="/workspace/result.txt", content="first"), AIMessage(content="done")]
    assert run(query="write", workspace=str(tmp_path)) == 0
    saved = SessionStore().list_sessions()[0]
    (tmp_path / "result.txt").write_text("user edit")
    responses[:] = [AIMessage(content="next turn")]
    assert run(resume_session=saved.session_id) == 0
    assert (tmp_path / "result.txt").read_text() == "user edit"
    assert run(resume_session=saved.session_id, query="continue") == 0
    count = len(requests)
    assert run(resume_session=saved.session_id, approval_mode="review") == 2
    assert len(requests) == count


def test_headless_clarification_survives_restart_and_interactive_answer(headless, monkeypatch):
    run, responses, requests = headless
    responses[:] = [call("load_tools", names=["ask_human"]), call("ask_human", question="Which calculator?"), AIMessage(content="done")]
    registry = ToolRegistry([ToolRegistry().get_spec("ask_human")])
    assert run(query="calculate", human_supervised=True, tool_registry=registry) == 3
    saved = SessionStore().list_sessions()[0]
    assert saved.status == "waiting_for_user"
    responses[:] = [AIMessage(content="done")]
    assert run(resume_session=saved.session_id, query="do not replace the question") == 3
    assert SessionStore().get_session(saved.session_id).status == "waiting_for_user"
    count = len(requests)
    assert commands.interactive_mode(workflow="main_agent", resume_session=saved.session_id) == 2
    assert len(requests) == count
    replies = iter(["EMT", "/quit"])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *a, **k: next(replies))
    commands.interactive_mode(workflow="main_agent", resume_session=saved.session_id, approval_mode="bypass")
    assert SessionStore().get_session(saved.session_id).status == "completed"


def test_headless_failure_is_not_retried(headless):
    run, responses, _ = headless
    responses[:] = [RuntimeError("model unavailable")]
    assert run(query="calculate") == 1
    saved = SessionStore().list_sessions()[0]
    assert saved.status == "failed"
    responses[:] = [AIMessage(content="must not run")]
    assert run(resume_session=saved.session_id, query="must not run") == 1
    assert SessionStore().get_session(saved.session_id).status == "failed"


@pytest.mark.parametrize("changes", [
    {"query": " "}, {"subagent_names": ["missing-worker"]},
    {"recursion_limit": 0}, {"workspace": ""},
])
def test_headless_configuration_errors_exit_two(headless, changes):
    run, _, _ = headless
    assert run(**{"query": "calculate", **changes}) == 2


@pytest.mark.parametrize("interactive", [False, True])
def test_cli_forwards_bypass_and_catalog(monkeypatch, interactive):
    received = {}
    monkeypatch.setattr(cli_main, "interactive_mode" if interactive else "run_headless_main_agent",
                        lambda **kwargs: received.update(kwargs))
    cli_main._handle_run(_run_args(interactive=interactive, dangerously_skip_approvals=True,
                                  query="calculate", local_tool_names=["calculator"]))
    assert received["approval_mode"] == "bypass"
    assert received["tool_registry"].names() == ("calculator",)


def test_cli_does_not_enable_bypass_from_toml(monkeypatch):
    monkeypatch.setattr(cli_main, "load_config", lambda _: {"dangerously_skip_approvals": True, "approval_mode": "bypass"})
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(_run_args(config="unused", query="calculate", dangerously_skip_approvals=False))
    assert exc.value.code == 2


@pytest.mark.parametrize("code", [1, 2, 3, 130])
def test_cli_propagates_headless_exit_status(monkeypatch, code):
    monkeypatch.setattr(cli_main, "run_headless_main_agent", lambda **kwargs: code)
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(_run_args(query="calculate", dangerously_skip_approvals=True))
    assert exc.value.code == code


def test_interactive_bypass_keeps_mode_across_model_changes(monkeypatch, tmp_path, model_loader):
    responses, _ = model_loader
    responses[:] = [AIMessage(content="done")]
    replies = iter(["/model test-two", "/workflow single_agent", "/config", "hello", "/quit"])
    monkeypatch.setattr(commands.Prompt, "ask", lambda *a, **k: next(replies))
    monkeypatch.setattr(commands.Confirm, "ask", lambda *a, **k: pytest.fail("bypass must not confirm"))
    with commands.console.capture() as output:
        commands.interactive_mode(model="test", workflow="main_agent", approval_mode="bypass",
                                  checkpoint_db=str(tmp_path / "checkpoints.db"), workspace=str(tmp_path))
    assert "Approval mode: bypass" in output.get()
    assert "Approval bypass requires" in output.get()
    saved = SessionStore().list_sessions()[0]
    assert SessionStore().get_session_metadata(saved.session_id)[1].graph_config.approval_mode == "bypass"


def test_legacy_flag_retains_headless_standalone_restriction(monkeypatch):
    monkeypatch.setattr(cli_main, "initialize_agent", lambda *a, **k: SimpleNamespace())
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(_run_args(interactive=True, workflow="deep_agent", deepagent_dangerously_skip_approvals=True))
    assert exc.value.code == 2


def test_cli_parser_executes_and_restores_headless_thread(headless):
    _, responses, requests = headless
    responses[:] = [AIMessage(content="done")]
    parser = cli_main.create_argument_parser()
    options = ["run", "-w", "main_agent", "--dangerously-skip-approvals", "--no-discover-skills"]
    # Use the fixture's temporary database rather than the CLI default.
    checkpoint_db = str(SessionStore().db_path) + ".checkpoints"
    options += ["--checkpoint-db", checkpoint_db]
    cli_main._handle_run(parser.parse_args([*options, "-q", "calculate"]))
    saved = SessionStore().list_sessions()[0]
    assert saved.status == "completed"
    cli_main._handle_run(parser.parse_args([*options, "--resume", saved.session_id]))
    count = len(requests)
    with pytest.raises(SystemExit) as exc:
        cli_main._handle_run(parser.parse_args(["run", "--interactive", "-w", "main_agent", "--resume", saved.session_id]))
    assert exc.value.code == 2
    assert len(requests) == count


def test_headless_refuses_to_upgrade_reviewed_session(headless):
    from chemgraph.cli.checkpoint_runtime import CheckpointRuntime

    run, responses, requests = headless
    responses[:] = [AIMessage(content="done")]
    checkpoint_db = str(SessionStore().db_path) + ".reviewed"
    with CheckpointRuntime() as runtime:
        agent = ChemGraph(workflow_type="main_agent", discover_skills=False,
                          checkpointer=runtime.open_sqlite(checkpoint_db))
        session = commands.create_main_agent_session(agent, checkpoint_db=checkpoint_db)
        assert runtime.run(lambda: session.run("hello")).status == "completed"
    count = len(requests)
    assert run(resume_session=session.thread_id) == 2
    assert len(requests) == count


def test_headless_cancellation_closes_checkpoint_runtime(headless, monkeypatch):
    from chemgraph.cli import headless as driver

    run, _, _ = headless
    runtimes = []
    original = driver.CheckpointRuntime
    def create_runtime():
        runtime = original()
        runtimes.append(runtime)
        return runtime
    def cancel(*args, **kwargs):
        raise KeyboardInterrupt
    monkeypatch.setattr(driver, "CheckpointRuntime", create_runtime)
    monkeypatch.setattr(commands, "run_main_agent_query", cancel)
    assert run(query="calculate") == 130
    assert len(runtimes) == 1 and runtimes[0]._closed
