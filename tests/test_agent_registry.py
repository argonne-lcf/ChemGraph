import pytest
from langgraph.checkpoint.memory import MemorySaver

from chemgraph.cli.commands import ALL_WORKFLOW_TYPES
from chemgraph.registry import (
    AgentRegistry,
    AgentSpec,
    DuplicateRegistryEntryError,
    RegistryUnavailableError,
    UnknownRegistryEntryError,
)
from chemgraph.tools.generic_tools import calculator


EXPECTED_WORKERS = (
    "single_agent",
    "multi_agent",
    "python_relp",
    "graspa",
    "mock_agent",
    "graspa_mcp",
    "rag_agent",
    "single_agent_xanes",
    "molecular_docking",
)


def _graspa_mcp_options():
    return {
        "executor_tools": [calculator],
        "analysis_tools": [calculator],
    }


def test_builtin_registry_contains_worker_graphs_not_main_agent():
    registry = AgentRegistry()

    assert registry.names() == EXPECTED_WORKERS
    assert set(registry.names()) == set(ALL_WORKFLOW_TYPES) - {"main_agent"}
    with pytest.raises(UnknownRegistryEntryError, match="Unknown worker agent"):
        registry.get_spec("main_agent")


def test_existing_workflow_aliases_resolve_to_canonical_workers():
    registry = AgentRegistry()

    assert registry.resolve_name("python_repl") == "python_relp"
    assert registry.resolve_name("graspa_agent") == "graspa"
    assert registry.get_spec("python_repl").name == "python_relp"


def test_registration_rejects_duplicate_names_and_aliases():
    registry = AgentRegistry(specs=())
    spec = AgentSpec("worker", "Worker", "tests.test_agent_registry:_factory")
    registry.register(spec)

    with pytest.raises(DuplicateRegistryEntryError, match="already registered"):
        registry.register(spec)
    with pytest.raises(DuplicateRegistryEntryError, match="already registered"):
        registry.register(
            AgentSpec(
                "another",
                "Another worker",
                "tests.test_agent_registry:_factory",
                aliases=("worker",),
            )
        )


def test_graspa_mcp_requires_both_external_tool_groups():
    registry = AgentRegistry()

    status = registry.availability("graspa_mcp")
    assert status.available is False
    assert status.issues == (
        "missing non-empty constructor argument 'executor_tools'",
        "missing non-empty constructor argument 'analysis_tools'",
    )

    with pytest.raises(RegistryUnavailableError, match="executor_tools"):
        registry.build("graspa_mcp", llm=object())


def test_all_workers_build_standalone_with_default_memory_checkpointer():
    registry = AgentRegistry()

    for name in registry.names():
        kwargs = _graspa_mcp_options() if name == "graspa_mcp" else {}
        graph = registry.build(
            name,
            llm=object(),
            require_available=False,
            **kwargs,
        )
        assert isinstance(graph.checkpointer, MemorySaver), name


def test_all_workers_adapt_to_parent_checkpointed_subagents():
    registry = AgentRegistry()
    workers = registry.as_subagents(
        registry.names(),
        llm=object(),
        options={"graspa_mcp": _graspa_mcp_options()},
        require_available=False,
    )

    assert tuple(worker["name"] for worker in workers) == EXPECTED_WORKERS
    assert all(worker["description"] for worker in workers)
    assert all(worker["runnable"].checkpointer is None for worker in workers)


def test_subagent_rejects_an_independent_checkpointer():
    with pytest.raises(ValueError, match="inherit the parent checkpointer"):
        AgentRegistry().as_subagent(
            "single_agent",
            llm=object(),
            require_available=False,
            checkpointer=MemorySaver(),
        )


def test_batch_validation_occurs_before_any_worker_is_built(monkeypatch):
    registry = AgentRegistry()
    calls = []
    monkeypatch.setattr(registry, "_get_constructor", lambda _spec: calls.append)

    with pytest.raises(RegistryUnavailableError, match="analysis_tools"):
        registry.as_subagents(
            ["single_agent", "graspa_mcp"],
            llm=object(),
            options={"graspa_mcp": {"executor_tools": [calculator]}},
        )
    assert calls == []


def test_alias_and_canonical_name_cannot_be_requested_together():
    with pytest.raises(DuplicateRegistryEntryError, match="more than once"):
        AgentRegistry().as_subagents(
            ["python_relp", "python_repl"],
            llm=object(),
            require_available=False,
        )
