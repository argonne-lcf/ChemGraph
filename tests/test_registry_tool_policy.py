"""Approval and completion policies for dynamically loaded native tools."""

import json
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
from deepagents.backends import LocalShellBackend
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.registry import ToolRegistry
from tests.test_registry_middleware import CatalogModel, call, names, outputs


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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("workspace", [False, True])
@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("operation", ["extract_output_json", "file_to_atomsdata", "load_document"])
async def test_host_readers_wait_for_approval(
    monkeypatch, tmp_path, operation, decision, workspace, asynchronous,
):
    registry = ToolRegistry()
    if operation == "load_document":
        from chemgraph.tools import rag_tools

        target = tmp_path / "document.txt"
        target.write_text("Host document contents")
        args = {"file_path": str(target)}
        splitter_module = ModuleType("langchain_text_splitters")
        splitter = Mock()
        splitter.create_documents.side_effect = lambda texts, metadatas: [
            Document(page_content=text, metadata=metadata)
            for text, metadata in zip(texts, metadatas)
        ]
        splitter_module.RecursiveCharacterTextSplitter = Mock(return_value=splitter)
        vector_module = ModuleType("langchain_community.vectorstores")
        vector_module.FAISS = Mock()
        embeddings = Mock(return_value=object())
        monkeypatch.setitem(sys.modules, splitter_module.__name__, splitter_module)
        monkeypatch.setitem(sys.modules, vector_module.__name__, vector_module)
        monkeypatch.setattr(rag_tools, "_get_embeddings", embeddings)
        monkeypatch.setattr(rag_tools, "_vector_stores", {})
        # Exercise the real reader with mocked optional dependencies.
        registry.register(rag_tools.load_document, replace=True)
    elif operation == "file_to_atomsdata":
        target = tmp_path / "hydrogen.xyz"
        target.write_text("2\nHost structure\nH 0 0 0\nH 0 0 1\n")
        args = {"fname": str(target)}
    else:
        target = tmp_path / "host.json"
        target.write_text('{"host_value": 42}')
        args = {"json_file": str(target)}
    reader = registry.get(operation)
    executed = Mock(wraps=reader.func)

    def record_read(*args, **kwargs):
        return executed(*args, **kwargs)

    monkeypatch.setattr(reader, "func", record_read)
    options = {}
    if workspace:
        root = tmp_path / "workspace"
        root.mkdir()
        options["backend"] = LocalShellBackend(root_dir=root, virtual_mode=True, env={})
    graph = construct_deep_agent_graph(
        CatalogModel(responses=[
            call("load_tools", names=[operation]), call(operation, **args), AIMessage(content="Done"),
        ]), tool_registry=registry, discover_skills=False, **options,
    )
    config = {"configurable": {"thread_id": "host-read"}}
    state = await invoke(graph, {"messages": [HumanMessage(content="Read the host file.")]}, config, asynchronous)
    assert state["__interrupt__"][0].value["action_requests"][0]["name"] == operation
    executed.assert_not_called()
    assert not any(message.name == operation for message in outputs(state))
    if operation == "load_document":
        assert not rag_tools._vector_stores
        embeddings.assert_not_called()
        vector_module.FAISS.from_documents.assert_not_called()
    state = await invoke(graph, Command(resume={"decisions": [{"type": decision}]}), config, asynchronous)
    assert "__interrupt__" not in state
    assert executed.call_count == (1 if decision == "approve" else 0)
    result = next(message for message in outputs(state) if message.name == operation)
    assert result.status == ("success" if decision == "approve" else "error")
    if operation == "load_document":
        if decision == "approve":
            embeddings.assert_called_once()
            vector_module.FAISS.from_documents.assert_called_once()
            chunks = vector_module.FAISS.from_documents.call_args.args[0]
            assert chunks[0].page_content == "Host document contents"
            assert rag_tools._vector_stores[str(target)] is vector_module.FAISS.from_documents.return_value
        else:
            assert not rag_tools._vector_stores
            embeddings.assert_not_called()
            vector_module.FAISS.from_documents.assert_not_called()


@pytest.mark.parametrize("mode,name", [("custom", "custom_writer")] + [
    (mode, name)
    for mode in ("attached", "disabled", "override")
    for name in ("run_ase", "load_document", "file_to_atomsdata", "extract_output_json")
])
def test_registry_reviews_preserve_explicit_policies(mode, name):
    executed = []

    @tool(name)
    def operation() -> str:
        """Record execution without accessing host files or a real calculator."""
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


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("local_direct,other_direct,placement", [
    (True, True, "none"), (True, True, "registry"),
    (True, False, "registry"), (True, True, "attached"),
    (True, False, "attached"), (False, True, "attached"),
])
async def test_direct_return_batches_and_next_turn(
    asynchronous, local_direct, other_direct, placement,
):
    executed = []

    @tool(return_direct=local_direct)
    def local_answer() -> str:
        """Return a local result."""
        executed.append("local")
        return "local result"

    @tool(return_direct=other_direct)
    def other_answer() -> str:
        """Return another result."""
        executed.append("other")
        return "other result"

    registry = ToolRegistry([])
    registry.register(local_answer)
    if placement == "registry":
        registry.register(other_answer)
    calls = [call("local_answer").tool_calls[0]]
    if placement != "none":
        calls += call("other_answer").tool_calls
    direct = local_direct and other_direct
    responses = [call("load_tools", names=list(registry.names())), AIMessage(content="", tool_calls=calls)]
    if not direct:
        responses.append(AIMessage(content="Processed the results"))
    responses.append(AIMessage(content="New turn"))
    model = CatalogModel(responses=responses)
    graph = construct_deep_agent_graph(
        model, tool_registry=registry, discover_skills=False,
        tools=[other_answer] if placement == "attached" else [],
    )
    config = {"configurable": {"thread_id": "direct"}}
    state = await invoke(graph, {"messages": [HumanMessage(content="Get results.")]}, config, asynchronous)
    assert sorted(executed) == (["local"] if placement == "none" else ["local", "other"])
    assert model.response_index == (2 if direct else 3)
    assert state["messages"][-1].type == ("tool" if direct else "ai")
    assert any(message.content == "local result" for message in outputs(state))
    snapshot = await graph.aget_state(config) if asynchronous else graph.get_state(config)
    assert snapshot.values["active_registry_tools"] == []
    state = await invoke(graph, {"messages": [HumanMessage(content="A new turn.")]}, config, asynchronous)
    assert state["messages"][-1].content == "New turn"
    assert "local_answer" not in names(model.schemas[-1])
    assert local_answer.return_direct == local_direct
    assert other_answer.return_direct == other_direct


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["approve", "reject", "validation", "error"])
async def test_direct_return_approval_reconstruction_and_errors(asynchronous, outcome):
    executed = []

    @tool(return_direct=True)
    def final_answer(value: int) -> str:
        """Return a final result, or a recoverable tool error."""
        executed.append(value)
        if value < 0:
            raise ToolException("calculation failed")
        return str(value)

    final_answer.handle_tool_error = True

    @tool(return_direct=True)
    def attached_answer() -> str:
        """Return an attached result to exercise static routing."""
        return "attached result"

    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "resume"}}

    def graph(responses):
        # A fresh registry on reconstruction must recover from checkpoint state.
        registry = ToolRegistry([])
        registry.register(final_answer)
        model = CatalogModel(responses=responses)
        agent = construct_deep_agent_graph(
            model, tool_registry=registry, tools=[attached_answer], discover_skills=False,
            checkpointer=saver, interrupt_on={"final_answer": True},
        )
        return agent, model

    value = "invalid" if outcome == "validation" else -1 if outcome == "error" else 42
    agent, _ = graph([
        call("load_tools", names=["final_answer"]),
        AIMessage(content="", tool_calls=[
            *call("final_answer", value=value).tool_calls, *call("attached_answer").tool_calls,
        ]),
    ])
    state = await invoke(agent, {"messages": [HumanMessage(content="Get results.")]}, config, asynchronous)
    assert state["__interrupt__"] and not executed
    isolated, _ = graph([AIMessage(content="Unrelated turn")])
    await invoke(isolated, {"messages": [HumanMessage(content="Hello")]}, {"configurable": {"thread_id": "other"}}, asynchronous)
    agent, model = graph([AIMessage(content="Recover from the rejected or failed call")])
    decision = "reject" if outcome == "reject" else "approve"
    state = await invoke(agent, Command(resume={"decisions": [{"type": decision}]}), config, asynchronous)
    assert "__interrupt__" not in state
    assert model.response_index == (0 if outcome == "approve" else 1)
    assert executed == ([42] if outcome == "approve" else [-1] if outcome == "error" else [])
    if outcome == "approve":
        assert any(message.content == "42" for message in outputs(state))
    else:
        assert any(message.status == "error" for message in outputs(state))
    snapshot = await agent.aget_state(config) if asynchronous else agent.get_state(config)
    assert snapshot.values["active_registry_tools"] == []
