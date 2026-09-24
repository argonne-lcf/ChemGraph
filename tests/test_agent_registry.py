import pytest
from langchain_core.messages import AIMessage
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
from tests.test_main_agent import _ScriptedChatModel


EXPECTED_WORKERS = (
    "single_agent",
    "deep_agent",
    "multi_agent",
    "python_relp",
    "graspa",
    "mock_agent",
    "graspa_mcp",
    "rag_agent",
    "single_agent_xanes",
    "molecular_docking",
    "single_agent_iri",
)


def _factory_v1(_llm):
    return "factory-v1"


def _factory_v2(_llm):
    return "factory-v2"


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
    assert registry.resolve_name("deepagent") == "deep_agent"
    assert registry.resolve_name("graspa_agent") == "graspa"
    assert registry.resolve_name("iri") == "single_agent_iri"
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


def test_replacement_retains_changes_aliases_and_invalidates_constructor():
    registry = AgentRegistry(specs=())
    registry.register(
        AgentSpec(
            "worker",
            "Worker v1",
            "tests.test_agent_registry:_factory_v1",
            aliases=("keep", "remove"),
        )
    )
    assert registry.build("worker", llm=object()) == "factory-v1"

    replacement = AgentSpec(
        "worker",
        "Worker v2",
        "tests.test_agent_registry:_factory_v2",
        aliases=("keep", "new"),
    )
    registry.register(replacement, replace=True)

    assert registry.get_spec("worker") is replacement
    assert registry.resolve_name("keep") == "worker"
    assert registry.resolve_name("new") == "worker"
    with pytest.raises(UnknownRegistryEntryError, match="Unknown worker agent"):
        registry.resolve_name("remove")
    assert registry.build("worker", llm=object()) == "factory-v2"


@pytest.mark.parametrize("foreign_identifier", ["other", "other_alias"])
def test_replacement_rejects_identifiers_owned_by_another_agent(
    foreign_identifier,
):
    original = AgentSpec(
        "worker",
        "Worker",
        "tests.test_agent_registry:_factory_v1",
        aliases=("worker_alias",),
    )
    other = AgentSpec(
        "other",
        "Other",
        "tests.test_agent_registry:_factory_v1",
        aliases=("other_alias",),
    )
    registry = AgentRegistry(specs=(original, other))

    with pytest.raises(DuplicateRegistryEntryError, match="already registered"):
        registry.register(
            AgentSpec(
                "worker",
                "Replacement",
                "tests.test_agent_registry:_factory_v2",
                aliases=("worker_alias", foreign_identifier),
            ),
            replace=True,
        )

    assert registry.specs() == (original, other)
    assert registry.resolve_name("worker_alias") == "worker"
    assert registry.resolve_name("other_alias") == "other"


def test_replacement_rejects_a_canonical_name_owned_as_an_alias():
    original = AgentSpec(
        "first",
        "First",
        "tests.test_agent_registry:_factory_v1",
        aliases=("second",),
    )
    registry = AgentRegistry(specs=(original,))

    with pytest.raises(DuplicateRegistryEntryError, match="already registered"):
        registry.register(
            AgentSpec(
                "second",
                "Second",
                "tests.test_agent_registry:_factory_v2",
            ),
            replace=True,
        )

    assert registry.specs() == (original,)
    assert registry.resolve_name("second") == "first"


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
            llm=_ScriptedChatModel(responses=[AIMessage(content="done")]),
            require_available=False,
            **kwargs,
        )
        assert isinstance(graph.checkpointer, MemorySaver), name


def test_all_workers_adapt_to_parent_checkpointed_subagents():
    registry = AgentRegistry()
    workers = registry.as_subagents(
        registry.names(),
        llm=_ScriptedChatModel(responses=[AIMessage(content="done")]),
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


def test_batch_options_are_bound_before_any_worker_is_built(monkeypatch):
    registry = AgentRegistry(
        specs=(
            AgentSpec("first", "First", "unused:first"),
            AgentSpec("second", "Second", "unused:second"),
        )
    )
    calls = []

    def constructor(_llm, checkpointer=None):
        calls.append(checkpointer)
        return object()

    monkeypatch.setattr(registry, "_get_constructor", lambda _spec: constructor)

    with pytest.raises(TypeError, match="Invalid options for agent 'second'"):
        registry.as_subagents(
            ["first", "second"],
            llm=object(),
            options={"second": {"unexpected": True}},
            require_available=False,
        )
    assert calls == []


def test_batch_rejects_an_independent_checkpointer_before_loading(monkeypatch):
    registry = AgentRegistry()
    calls = []
    monkeypatch.setattr(registry, "_get_constructor", lambda spec: calls.append(spec))

    with pytest.raises(ValueError, match="inherit the parent checkpointer"):
        registry.as_subagents(
            ["single_agent", "python_relp"],
            llm=object(),
            options={"python_relp": {"checkpointer": MemorySaver()}},
            require_available=False,
        )
    assert calls == []


def test_alias_and_canonical_name_cannot_be_requested_together():
    with pytest.raises(DuplicateRegistryEntryError, match="more than once"):
        AgentRegistry().as_subagents(
            ["python_relp", "python_repl"],
            llm=object(),
            require_available=False,
        )
