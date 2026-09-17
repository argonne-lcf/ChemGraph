"""Approval and completion policies for dynamically loaded native tools."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command

from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.registry import ToolRegistry
from tests.test_registry_middleware import CatalogModel, call


async def invoke(graph, value, config, asynchronous):
    if asynchronous:
        return await graph.ainvoke(value, config)
    return graph.invoke(value, config)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("operation", ["run_ase", "generate_html"])
async def test_artifact_writers_wait_for_approval(tmp_path, operation, decision, asynchronous):
    from chemgraph.schemas.ase_input import ASEInputSchema
    from chemgraph.tools.ase_core import run_ase_core

    structure = tmp_path / "hydrogen.xyz"
    structure.write_text("2\nEMT test\nH 0 0 0\nH 0 0 1\n")
    result_file = tmp_path / "result.json"
    params = {
        "input_structure_file": str(structure), "output_results_file": str(result_file),
        "driver": "energy", "calculator": {"calculator_type": "emt"},
    }
    if operation == "generate_html":
        assert run_ase_core(ASEInputSchema.model_validate(params))["status"] == "success"
        target = tmp_path / "report.html"
        args = {"results_json_path": str(result_file), "output_path": str(target)}
    else:
        target = result_file
        args = {"params": params}
    target.write_text("existing user data")
    model = CatalogModel(responses=[
        call("load_tools", names=[operation]), call(operation, **args), AIMessage(content="Done"),
    ])
    graph = construct_deep_agent_graph(model, tool_registry=ToolRegistry(), discover_skills=False)
    config = {"configurable": {"thread_id": "artifact"}}
    state = await invoke(graph, {"messages": [HumanMessage(content="Create the artifact.")]}, config, asynchronous)
    assert state["__interrupt__"]
    assert target.read_text() == "existing user data"
    state = await invoke(graph, Command(resume={"decisions": [{"type": decision}]}), config, asynchronous)
    assert "__interrupt__" not in state
    if decision == "reject":
        assert target.read_text() == "existing user data"
    elif operation == "run_ase":
        assert isinstance(json.loads(target.read_text())["potential_energy"], float)
    else:
        assert "<html" in target.read_text().lower()


@pytest.mark.parametrize("operation", [
    "run_docking", "run_graspa", "run_xanes", "fetch_xanes_data", "plot_xanes_data",
])
@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_optional_operations_wait_for_approval(operation, decision):
    executed = []

    @tool(operation)
    def operation_tool() -> str:
        """Stand in for an optional calculation or artifact writer."""
        executed.append(operation)
        return "completed"

    registry = ToolRegistry([])
    registry.register(operation_tool)
    graph = construct_deep_agent_graph(
        CatalogModel(responses=[
            call("load_tools", names=[operation]), call(operation), AIMessage(content="Done"),
        ]), tool_registry=registry, discover_skills=False,
    )
    config = {"configurable": {"thread_id": operation}}
    state = graph.invoke({"messages": [HumanMessage(content="Run it.")]}, config)
    assert state["__interrupt__"] and not executed
    graph.invoke(Command(resume={"decisions": [{"type": decision}]}), config)
    assert executed == ([operation] if decision == "approve" else [])


@pytest.mark.parametrize("mode", ["attached", "disabled", "override", "custom"])
def test_registry_reviews_preserve_explicit_policies(mode):
    executed = []
    name = "custom_writer" if mode == "custom" else "run_ase"

    @tool(name)
    def operation() -> str:
        """Record execution without a real calculator."""
        executed.append(name)
        return "completed"

    registry = ToolRegistry([])
    options = {}
    if mode == "attached":
        options["tools"] = [operation]
        # An unrelated registry must not add reviews to the attached tool.
        registry.register(ToolRegistry().get_spec("calculator"))
    else:
        registry.register(operation)
    if mode == "disabled":
        options["interrupt_on"] = None
    elif mode == "override":
        options["interrupt_on"] = {name: False}
    responses = [] if mode == "attached" else [call("load_tools", names=[name])]
    graph = construct_deep_agent_graph(
        CatalogModel(responses=[*responses, call(name), AIMessage(content="Done")]),
        tool_registry=registry, discover_skills=False, **options,
    )
    state = graph.invoke(
        {"messages": [HumanMessage(content="Run it.")]},
        {"configurable": {"thread_id": mode}},
    )
    assert "__interrupt__" not in state and executed == [name]
