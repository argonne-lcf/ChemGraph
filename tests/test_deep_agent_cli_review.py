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
