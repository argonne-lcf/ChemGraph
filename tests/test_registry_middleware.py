"""Provider-independent discovery, native tool execution, and turn isolation."""

import json
import sys
from types import ModuleType

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.registry.middleware import RegistryToolsMiddleware
from chemgraph.registry.tools import RuntimeRequirement, ToolRegistry, ToolSpec
from tests.test_main_agent import _ScriptedChatModel


class CatalogModel(_ScriptedChatModel):
    schemas: list = Field(default_factory=list)

    def _generate(self, *args, **kwargs):
        self.schemas.append([convert_to_openai_tool(t) for t in self.bound_tools])
        return super()._generate(*args, **kwargs)


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


def names(schema):
    return {item["function"]["name"] for item in schema}


def outputs(state):
    return [m for m in state["messages"] if m.type == "tool"]


@pytest.fixture
def catalog(monkeypatch):
    executed = []

    @tool
    def double(value: int) -> int:
        """Double an integer."""
        executed.append(value)
        if value < 0:
            raise ValueError("negative value")
        return value * 2

    @tool
    def triple(value: int) -> int:
        """Triple an integer."""
        return value * 3

    module = ModuleType("test_lazy_chemgraph_tools")
    module.double, module.triple = double, triple
    monkeypatch.setitem(sys.modules, module.__name__, module)
    registry = ToolRegistry(
        [
            ToolSpec(
                name,
                f"{name} an integer",
                f"{module.__name__}:{name}",
                frozenset({"math"}),
            )
            for name in ("double", "triple")
        ]
        + [
            ToolSpec(
                "unavailable",
                "Missing optional calculator",
                "absent.module:tool",
                requirements=(
                    RuntimeRequirement("module", "chemgraph_absent_dependency"),
                ),
            )
        ]
    )
    imports = []
    original = registry.get

    def get(name, **kwargs):
        if name not in registry._tools:
            imports.append(name)
        return original(name, **kwargs)

    monkeypatch.setattr(registry, "get", get)
    return registry, executed, imports


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_lazy_binding_execution_and_turn_cleanup(catalog, asynchronous):
    registry, executed, imports = catalog
    model = CatalogModel(
        responses=[
            call("search_tools", query="math"),
            call("load_tools", names=["double"]),
            call("double", value=4),
            AIMessage(content="Eight."),
            call("double", value=5),
            AIMessage(content="Load it first."),
        ]
    )
    graph = construct_deep_agent_graph(
        model, tool_registry=registry, discover_skills=False
    )
    assert imports == []
    config = {"configurable": {"thread_id": "one"}}
    invoke = graph.ainvoke if asynchronous else graph.invoke
    result = invoke({"messages": [HumanMessage(content="Double four.")]}, config)
    if asynchronous:
        result = await result
    assert executed == [4] and imports == ["double"]
    assert "double" not in names(model.schemas[0])
    assert "double" not in names(model.schemas[1])
    assert "double" in names(model.schemas[2])
    assert "triple" not in names(model.schemas[2])
    assert any(m.content == "8" for m in outputs(result))
    assert graph.get_state(config).values["active_registry_tools"] == []
    result = invoke({"messages": [HumanMessage(content="A new turn.")]}, config)
    if asynchronous:
        result = await result
    assert "double" not in names(model.schemas[4])
    assert executed == [4]
    assert outputs(result)[-1].status == "error"


def test_failed_loading_preserves_selection_and_replacement(catalog):
    registry, executed, _ = catalog
    model = CatalogModel(
        responses=[
            call("load_tools", names=["double"]),
            call("load_tools", names=["triple", "unavailable"]),
            call("double", value=3),
            call("load_tools", names=["unknown"]),
            call("load_tools", names=["triple"]),
            call("triple", value=4),
            call("load_tools", names=[]),
            AIMessage(content="Done."),
        ]
    )
    result = construct_deep_agent_graph(
        model,
        tool_registry=registry,
        discover_skills=False,
    ).invoke(
        {"messages": [HumanMessage(content="Use tools.")]},
        {"configurable": {"thread_id": "test"}},
    )
    assert executed == [3]
    assert "double" in names(model.schemas[2])
    assert "triple" not in names(model.schemas[2])
    assert "double" not in names(model.schemas[5])
    assert "triple" in names(model.schemas[5])
    assert not {"double", "triple"} & names(model.schemas[7])
    assert len([m for m in outputs(result) if m.status == "error"]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_selection_survives_reconstruction_and_approval(
    catalog, asynchronous, decision
):
    registry, executed, _ = catalog
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "approval"}}

    def graph(responses):
        return construct_deep_agent_graph(
            CatalogModel(responses=responses),
            tool_registry=registry,
            discover_skills=False,
            checkpointer=saver,
            interrupt_on={"double": True},
        )

    agent = graph([call("load_tools", names=["double"]), call("double", value=9)])
    if asynchronous:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Double nine.")]}, config
        )
    else:
        result = agent.invoke(
            {"messages": [HumanMessage(content="Double nine.")]}, config
        )
    assert result["__interrupt__"] and not executed
    assert agent.get_state(config).values["active_registry_tools"] == ["double"]
    # Another conversation starts with no selection, even using the same registry.
    isolated = graph([AIMessage(content="Hello.")])
    isolated.invoke(
        {"messages": [HumanMessage(content="Hello.")]},
        {"configurable": {"thread_id": "other"}},
    )
    assert isolated.get_state(config).values["active_registry_tools"] == ["double"]
    agent = graph([AIMessage(content="Done.")])
    resume = Command(resume={"decisions": [{"type": decision}]})
    result = (
        await agent.ainvoke(resume, config)
        if asynchronous
        else agent.invoke(resume, config)
    )
    assert "__interrupt__" not in result
    assert executed == ([9] if decision == "approve" else [])
    assert agent.get_state(config).values["active_registry_tools"] == []


def test_collisions_and_search_without_imports(catalog):
    registry, _, imports = catalog
    search = RegistryToolsMiddleware(registry).tools[0]
    assert search.invoke({"query": "math", "limit": 1}) == {
        "tools": [{"name": "double", "description": "double an integer"}],
    }
    assert "error" in search.invoke({"query": "", "limit": 100})
    assert imports == []
    with pytest.raises(ValueError, match="conflict"):
        construct_deep_agent_graph(
            object(), tool_registry=registry, tools=[registry.get("double")]
        )
    with pytest.raises(ValueError, match="conflict"):
        RegistryToolsMiddleware(
            ToolRegistry([ToolSpec("execute", "bad", "unused:tool")])
        )


def test_parallel_load_requests_return_errors(catalog):
    registry, _, imports = catalog
    model = CatalogModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "load_tools", "args": {"names": [name]}, "id": name}
                    for name in ("double", "triple")
                ],
            ),
            AIMessage(content="Retry sequentially."),
        ]
    )
    result = construct_deep_agent_graph(
        model, tool_registry=registry, discover_skills=False
    ).invoke(
        {"messages": [HumanMessage(content="Load tools.")]},
        {"configurable": {"thread_id": "parallel"}},
    )
    assert imports == []
    assert all(m.status == "error" for m in outputs(result))


def test_registry_schemas_stay_out_of_initial_model_request():
    registry = ToolRegistry()
    model = CatalogModel(responses=[AIMessage(content="Hello.")])
    construct_deep_agent_graph(
        model, tool_registry=registry, discover_skills=False
    ).invoke(
        {"messages": [HumanMessage(content="Hello.")]},
        {"configurable": {"thread_id": "schema"}},
    )
    assert not set(registry.names()) & names(model.schemas[0])
    assert registry._tools == {}
    # Compare actual serialized schemas, without pretending bytes are token counts.
    discovery = [
        s
        for s in model.schemas[0]
        if s["function"]["name"] in {"search_tools", "load_tools"}
    ]
    eager = [
        convert_to_openai_tool(t)
        for t in registry.resolve(
            [
                "smiles_to_coordinate_file",
                "file_to_atomsdata",
                "extract_output_json",
                "run_ase",
            ]
        )
    ]
    assert len(json.dumps(discovery)) < len(json.dumps(eager))


def test_native_validation_and_always_attached_tools(catalog):
    registry, executed, _ = catalog

    @tool
    def attached() -> str:
        """An always-attached tool, as with an MCP adapter."""
        return "attached result"

    model = CatalogModel(
        responses=[
            call("attached"),
            call("load_tools", names=["double"]),
            call("double", value="not an integer"),
            AIMessage(content="Invalid arguments."),
        ]
    )
    result = construct_deep_agent_graph(
        model,
        tools=[attached],
        tool_registry=registry,
        discover_skills=False,
    ).invoke(
        {"messages": [HumanMessage(content="Use both tools.")]},
        {"configurable": {"thread_id": "native"}},
    )
    assert not executed
    assert all("attached" in names(schema) for schema in model.schemas)
    assert outputs(result)[0].content == "attached result"
    assert outputs(result)[-1].status == "error"


def test_new_request_clears_selection_after_tool_failure(catalog):
    registry, _, _ = catalog
    model = CatalogModel(
        responses=[
            call("load_tools", names=["double"]),
            call("double", value=-1),
            AIMessage(content="New request."),
        ]
    )
    graph = construct_deep_agent_graph(
        model, tool_registry=registry, discover_skills=False
    )
    config = {"configurable": {"thread_id": "failed"}}
    with pytest.raises(ValueError, match="negative value"):
        graph.invoke({"messages": [HumanMessage(content="Fail.")]}, config)
    graph.invoke({"messages": [HumanMessage(content="Start again.")]}, config)
    assert "double" not in names(model.schemas[-1])


def test_discovery_names_cannot_collide_with_attached_tools(catalog):
    registry, _, _ = catalog

    @tool
    def search_tools(query: str) -> str:
        """An incompatible MCP tool name."""
        return query

    with pytest.raises(ValueError, match="conflict"):
        construct_deep_agent_graph(
            object(), tools=[search_tools], tool_registry=registry
        )
