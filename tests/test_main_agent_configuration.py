"""Main-agent Python options and reconstruction metadata."""

from dataclasses import replace

import pytest
from deepagents.backends import LocalShellBackend
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

from chemgraph.agent.llm_agent import ChemGraph, PromptConfig
from chemgraph.cli.commands import _main_agent_options
from chemgraph.graphs.workspace import create_cli_workspace_backend
from chemgraph.memory.schemas import MainAgentGraphConfig
from chemgraph.models.endpoints import PreparedModel
from chemgraph.registry.tools import ToolRegistry
from tests.test_main_agent import _ScriptedChatModel


@pytest.fixture
def api(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", lambda **kwargs: (
        _ScriptedChatModel(responses=[AIMessage(content="done")]),
        PreparedModel(endpoint_name="test", protocol="openai_compatible", client_kwargs={}),
    ))
    monkeypatch.setattr("chemgraph.agent.llm_agent.construct_main_agent_graph",
                        lambda *args, **kwargs: captured.update(kwargs))

    def create(**kwargs):
        return ChemGraph(workflow_type="main_agent", enable_memory=False,
                         log_dir=str(tmp_path), **kwargs)
    return create, captured


def test_python_options_are_applied_and_reconstructed(api, tmp_path):
    create, captured = api
    registry = ToolRegistry([ToolRegistry().get_spec("calculator")])
    agent = create(backend=create_cli_workspace_backend(tmp_path),
                   skills=["/workspace/skills/"], skill_dirs=[str(tmp_path)],
                   discover_skills=False, tool_registry=registry,
                   prompts=PromptConfig(main_agent="custom main prompt"))
    assert captured["backend"] is agent.backend
    assert captured["system_prompt"] == "custom main prompt"
    assert captured["tool_registry"] is registry
    assert captured["skills"] == ("/workspace/skills/",)
    config = agent.main_agent_metadata.graph_config
    restored = _main_agent_options(config)
    assert config.graph_schema_version == 3
    assert restored["workspace"] == str(tmp_path.resolve())
    assert restored["skill_dirs"] == (str(tmp_path.resolve()),)
    assert restored["tool_registry"].names() == ("calculator",)
    assert restored["main_agent_prompt"] == "custom main prompt"
    assert restored["discover_skills"] is False


def test_catalog_defaults_filter_attached_and_interactive_tools(api):
    create, captured = api
    @tool
    def calculator(value: str) -> str:
        """Custom calculator."""
        return value
    agent = create(tools=[calculator])
    assert "calculator" not in captured["tool_registry"].names()
    assert "ask_human" not in captured["tool_registry"].names()
    assert not agent.main_agent_metadata.graph_config.cli_restorable
    create(human_supervised=True)
    assert "ask_human" in captured["tool_registry"].names()
    create(tool_registry=ToolRegistry([]))
    assert captured["tool_registry"].names() == ()


def test_named_workers_stay_lazy_and_accept_python_overrides(api, monkeypatch, tmp_path):
    create, captured = api
    monkeypatch.setattr("chemgraph.registry.agents.AgentRegistry.as_subagents",
                        lambda *args, **kwargs: pytest.fail("workers must remain lazy"))
    backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=True, env={})
    agent = create(backend=backend, discover_skills=False,
                   subagent_names=["deepagent", "single_agent"],
                   subagent_options={"deepagent": {"system_prompt": "worker prompt"}})
    assert captured["agent_registry"].names() == ("deep_agent", "single_agent")
    assert captured["agent_options"]["deep_agent"]["system_prompt"] == "worker prompt"
    assert agent.main_agent_metadata.graph_config.subagent_names == ("deep_agent", "single_agent")
    with pytest.raises(ValueError, match="Python"):
        _main_agent_options(agent.main_agent_metadata.graph_config)


@pytest.mark.parametrize("kwargs", [
    {"subagent_names": "deep_agent"},
    {"subagent_names": ["deepagent", "deep_agent"]},
    {"subagent_names": ["deep_agent"], "enable_deepagent": True},
    {"subagent_options": {"single_agent": []}},
    {"subagent_names": ["single_agent"], "subagent_options": {"deep_agent": {}}},
])
def test_invalid_worker_selections_fail_before_model_loading(monkeypatch, kwargs):
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared",
                        lambda **kwargs: pytest.fail("validation must precede model loading"))
    from chemgraph.registry.tools import RegistryError
    with pytest.raises((ValueError, TypeError, RegistryError)):
        ChemGraph(workflow_type="main_agent", **kwargs)


@pytest.mark.parametrize("workflow", ["python_relp", "python_repl"])
def test_removed_workflow_migration_precedes_workspace_validation(monkeypatch, workflow):
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared",
                        lambda **kwargs: pytest.fail("must reject before loading the model"))
    with pytest.raises(ValueError, match="has been removed"):
        ChemGraph(workflow_type=workflow, backend=object())


def test_graph_uses_the_fingerprinted_effective_policy(api, monkeypatch):
    from chemgraph.memory.graph_config import fingerprint

    create, captured = api
    payloads = []
    def record(payload):
        payloads.append(payload)
        return fingerprint(payload)
    monkeypatch.setattr("chemgraph.agent.llm_agent.fingerprint", record)
    create()
    policy = captured["interrupt_on"]
    assert payloads[0]["review_policy"]["interrupt_on"] == policy
    assert policy["run_ase"] == policy["execute"] == {"allowed_decisions": ["approve", "reject"]}


def test_custom_catalog_prose_and_tool_schemas_keep_full_identities(api):
    create, _ = api
    spec = ToolRegistry().get_spec("calculator")
    first = create(tool_registry=ToolRegistry([replace(spec, description="custom one")]),
                   configuration_id="custom-v1").main_agent_metadata.graph_config
    second = create(tool_registry=ToolRegistry([replace(spec, description="custom two")]),
                    configuration_id="custom-v1").main_agent_metadata.graph_config
    assert first.topology_fingerprint != second.topology_fingerprint
    assert first.requires_configuration_id and not first.cli_restorable

    @tool("custom")
    def integer_tool(value: int) -> str:
        """Custom tool."""
        return str(value)
    @tool("custom")
    def string_tool(value: str) -> str:
        """Custom tool."""
        return value
    assert (create(tools=[integer_tool], configuration_id="custom-v1").main_agent_metadata.graph_config.topology_fingerprint
            != create(tools=[string_tool], configuration_id="custom-v1").main_agent_metadata.graph_config.topology_fingerprint)


def test_legacy_metadata_is_identified_and_not_restored_by_cli():
    legacy = MainAgentGraphConfig.model_validate({"model_name": "old"})
    assert legacy.graph_schema_version == 1
    with pytest.raises(ValueError, match="Start a new session"):
        _main_agent_options(legacy)


def test_custom_shell_requires_identity_and_never_persists_environment(api, tmp_path):
    create, _ = api
    def config(**changes):
        backend = LocalShellBackend(root_dir=tmp_path, env={"SECRET_TEST": "do-not-persist"}, **changes)
        return create(backend=backend, configuration_id="workspace-v1").main_agent_metadata.graph_config
    first, second = config(), config()
    assert first.topology_fingerprint == second.topology_fingerprint
    assert first.requires_configuration_id and not first.cli_restorable
    assert "do-not-persist" not in first.model_dump_json()
    assert first.topology_fingerprint != config(timeout=19).topology_fingerprint
    with pytest.raises(ValueError, match="Python"):
        _main_agent_options(first)


def test_main_and_legacy_worker_workspace_identities_are_independent(api, tmp_path):
    create, _ = api
    other = tmp_path / "other"
    other.mkdir()
    def config(main, worker):
        return create(backend=create_cli_workspace_backend(main), enable_deepagent=True,
                      deepagent_backend=create_cli_workspace_backend(worker),
                      discover_skills=False, deepagent_discover_skills=False).main_agent_metadata.graph_config
    original = config(tmp_path, tmp_path)
    assert original.cli_restorable
    assert len({config(main, worker).topology_fingerprint
                for main, worker in ((tmp_path, tmp_path), (other, tmp_path), (tmp_path, other))}) == 3


def test_equivalent_worker_objects_have_stable_identities(api, tmp_path):
    create, _ = api
    def config(identity, timeout=120):
        return create(subagent_names=["deep_agent"], configuration_id=identity,
                      subagent_options={"deep_agent": {
                          "backend": LocalShellBackend(root_dir=tmp_path, env={}, timeout=timeout),
                      }}).main_agent_metadata.graph_config
    first = config("worker-v1")
    assert first.topology_fingerprint == config("worker-v1").topology_fingerprint
    assert first.topology_fingerprint != config("worker-v2").topology_fingerprint
    assert first.topology_fingerprint != config("worker-v1", timeout=15).topology_fingerprint
    assert first.requires_configuration_id and not first.cli_restorable


def test_cli_backend_roundtrip_and_mutation_detection(api, tmp_path):
    create, _ = api
    backend = create_cli_workspace_backend(tmp_path)
    original = create(backend=backend, configuration_id="catalog-v1").main_agent_metadata.graph_config
    options = _main_agent_options(original)
    options["backend"] = create_cli_workspace_backend(options.pop("workspace"))
    options["prompts"] = PromptConfig(main_agent=options.pop("main_agent_prompt"))
    reconstructed = create(**options).main_agent_metadata.graph_config
    assert reconstructed.topology_fingerprint == original.topology_fingerprint
    backend._env["CUSTOM_TEST_SETTING"] = "changed"
    assert not create(backend=backend).main_agent_metadata.graph_config.cli_restorable


@pytest.mark.parametrize("prompts", [PromptConfig(system="custom"), PromptConfig(formatter="custom"),
                                     PromptConfig(report="custom"), PromptConfig(deepagent="custom")])
def test_unpersisted_worker_prompts_require_python_reconstruction(api, prompts):
    create, _ = api
    agent = create(enable_deepagent=True, prompts=prompts)
    assert not agent.main_agent_metadata.graph_config.cli_restorable


@pytest.mark.parametrize("kwargs", [{"configuration_id": ""}, {"configuration_id": 1},
                                   {"subagent_names": [None]}, {"subagent_options": []}])
def test_malformed_configuration_fails_before_model_loading(monkeypatch, kwargs):
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared",
                        lambda **kwargs: pytest.fail("must validate before loading"))
    with pytest.raises((TypeError, ValueError)):
        ChemGraph(workflow_type="main_agent", **kwargs)


def test_state_only_legacy_worker_is_not_recreated_as_cli_host_shell(api):
    create, _ = api
    config = create(enable_deepagent=True).main_agent_metadata.graph_config
    assert not config.cli_restorable
    assert not config.requires_configuration_id
    with pytest.raises(ValueError, match="Python"):
        _main_agent_options(config)


def test_unavailable_workers_can_be_discovered_without_building(api):
    create, captured = api
    agent = create(subagent_names=["graspa_mcp"])
    assert captured["agent_registry"].names() == ("graspa_mcp",)
    status = agent.agent_registry.availability("graspa_mcp")
    assert not status.available and "executor_tools" in str(status.issues)


def test_worker_skill_directories_are_canonicalized_before_fingerprinting(api, monkeypatch, tmp_path):
    create, _ = api
    skill_root = tmp_path / "skills"
    skill_root.mkdir()
    monkeypatch.chdir(tmp_path)
    agent = create(subagent_names=["deep_agent"],
                   subagent_options={"deep_agent": {"skill_dirs": ["skills"]}})
    assert agent.subagent_options["deep_agent"]["skill_dirs"] == (str(skill_root.resolve()),)
