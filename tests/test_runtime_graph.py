"""Scoped main-agent graphs: review, denial and journaled execution with real EMT."""

import asyncio
import json
import shutil
import time
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from chemgraph.agent.approvals import normalize_approval_mode
from chemgraph.graphs.main_agent import construct_main_agent_graph
from chemgraph.registry import ToolRegistry
from chemgraph.runtime import (
    FileOperationStore, OperationJournal, ScopedExecutionError, artifact_file, result_artifacts,
    run_context,
)
from chemgraph.runtime import journal as journal_module
from chemgraph.runtime.artifact_audit import audit_journal
from chemgraph.runtime.policy import prepare
from chemgraph.tools.ase_tools import run_ase
from chemgraph.tools.cheminformatics_tools import smiles_to_coordinate_file
from tests.runtime_helpers import ai, approve, make_context, run_ase_call
from tests.test_main_agent import _ScriptedChatModel
from tests.test_main_agent_execution_context import model_loader as model_loader


def scoped_graph(responses, *, registry=False, saver=None):
    catalog = ToolRegistry(spec for spec in ToolRegistry().specs() if spec.name == "run_ase")
    options = ({"tool_registry": catalog} if registry else
               {"main_tools": [run_ase, smiles_to_coordinate_file], "tool_registry": ToolRegistry([])})
    return construct_main_agent_graph(
        _ScriptedChatModel(responses=responses), checkpointer=saver or InMemorySaver(),
        discover_skills=False, interrupt_on=None, scoped=True, **options,
    )


async def run(graph, context, value, thread="one"):
    with run_context(context):
        return await graph.ainvoke(value, {"configurable": {"thread_id": thread}})


async def decide_pending(context, state, *, approve_all=True, message=None):
    pending = [item.value for item in state["__interrupt__"]]
    assert all(item["kind"] == "approval" for item in pending)
    for item in pending:
        record = await context.journal.get(item["operation_id"])
        assert record["prepared"]["digest"] == item["digest"]
        if approve_all:
            await approve(context, record)
        else:
            await context.journal.decide(record["id"], digest=item["digest"], approve=False,
                                         decided_by="operator", message=message)
    return pending


def tool_messages(state):
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


@pytest.mark.asyncio
@pytest.mark.parametrize("registry", [False, True])
async def test_reviewed_energy_runs_once_with_artifacts(tmp_path, registry, monkeypatch):
    from ase.calculators.emt import EMT

    context = make_context(tmp_path)
    responses = [ai(run_ase_call()), AIMessage(content="Done")]
    if registry:
        responses.insert(0, ai({"name": "load_tools", "args": {"names": ["run_ase"]}, "id": "load"}))
    graph = scoped_graph(responses, registry=registry)
    paused = await run(graph, context, {"messages": [HumanMessage(content="Energy of H2")]})
    pending = await decide_pending(context, paused)
    assert pending[0]["tool_call_id"] == "energy" and pending[0]["tool"] == "run_ase"
    assert not (context.workspace / "energy.json").exists()

    # The resume value carries no authority; the journal decision does.
    state = await run(graph, context, Command(resume="anything"))
    assert state["messages"][-1].content == "Done"
    artifacts = result_artifacts(state, context.task_id)
    assert [a["source_path"] for a in artifacts] == ["energy.json"]
    saved = json.loads(artifact_file(context.workspace, artifacts[0]).read_text())
    assert saved["simulation_input"]["calculator"]["calculator_type"] == "emt"
    assert saved["input_structure_file"].startswith(str(context.workspace / ".chemgraph-inputs"))
    record = await context.journal.get(pending[0]["operation_id"])
    assert record["state"] == "finalized"

    def forbidden(*args, **kwargs):
        pytest.fail("Audit attempted computation or message decoding")
    monkeypatch.setattr(EMT, "calculate", forbidden)
    moved = tmp_path / "relocated"
    shutil.copytree(context.workspace, moved)
    (moved / "h2.xyz").unlink()
    (moved / "energy.json").unlink()
    report = audit_journal(workspace=moved, journal=tmp_path / "journal",
                           thread_id=context.thread_id, include_inputs=True)
    assert report["outcome"] == "verified" and report["counts"]["checked"] == 2


@pytest.mark.asyncio
async def test_identical_calls_are_reviewed_and_executed_separately(tmp_path):
    context = make_context(tmp_path)
    # Stage the shared input once before exercising independent approvals.
    prepare(context, run_ase_call())
    graph = scoped_graph([
        ai(run_ase_call("a", output="a.json"), run_ase_call("b", output="a.json")),
        AIMessage(content="Done"),
    ])
    paused = await run(graph, context, {"messages": [HumanMessage(content="Twice")]})
    pending = await decide_pending(context, paused)
    assert sorted(item["tool_call_id"] for item in pending) == ["a", "b"], await context.journal.store.records()
    state = await run(graph, context, Command(resume={i.id: "ok" for i in paused["__interrupt__"]}))
    artifacts = result_artifacts(state, context.task_id)
    assert sorted(a["tool_call_id"] for a in artifacts) == ["a", "b"]


@pytest.mark.asyncio
async def test_rejection_feedback_allows_a_revised_proposal(tmp_path):
    context = make_context(tmp_path)
    graph = scoped_graph([
        ai(run_ase_call()), ai(run_ase_call("revised", output="revised.json")),
        AIMessage(content="Done with the revised settings"),
    ])
    paused = await run(graph, context, {"messages": [HumanMessage(content="Energy")]})
    # Resuming without a recorded decision pauses again for the same operation.
    again = await run(graph, context, Command(resume="approve"))
    assert again["__interrupt__"][0].value["operation_id"] == paused["__interrupt__"][0].value["operation_id"]
    await decide_pending(context, again, approve_all=False, message="Use a different output name.")
    revised = await run(graph, context, Command(resume="decided"))
    rejection = tool_messages(revised)[-1]
    assert rejection.status == "error" and "Operator feedback: Use a different output name." in rejection.content
    assert "Do not retry" not in rejection.content
    assert not (context.workspace / "energy.json").exists()
    # The revised call is a new operation with its own review.
    assert revised["__interrupt__"][0].value["tool_call_id"] == "revised"
    await decide_pending(context, revised)
    state = await run(graph, context, Command(resume="decided"))
    assert [a["source_path"] for a in result_artifacts(state, context.task_id)] == ["revised.json"]


@pytest.mark.asyncio
async def test_rejection_without_feedback_discourages_retry(tmp_path):
    context = make_context(tmp_path)
    graph = scoped_graph([ai(run_ase_call()), AIMessage(content="Understood")])
    paused = await run(graph, context, {"messages": [HumanMessage(content="Energy")]})
    await decide_pending(context, paused, approve_all=False)
    state = await run(graph, context, Command(resume="decided"))
    assert "Do not retry it unless the user asks." in tool_messages(state)[-1].content
    assert state["messages"][-1].content == "Understood"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["expire", "reject"])
async def test_decision_change_before_invocation_returns_to_review(tmp_path, monkeypatch, change):
    context = make_context(tmp_path)
    graph = scoped_graph([ai(run_ase_call()), AIMessage(content="Understood")])
    paused = await run(graph, context, {"messages": [HumanMessage(content="Energy")]})
    record = await context.journal.get(paused["__interrupt__"][0].value["operation_id"])
    await approve(context, record, expires_at=time.time() + 60)
    validate, clock = journal_module.validate, time.time

    def change_meanwhile(*args):
        validate(*args)
        if change == "expire":
            monkeypatch.setattr(journal_module.time, "time", lambda: clock() + 120)
        else:
            operator = OperationJournal(FileOperationStore(tmp_path / "journal"))
            asyncio.run(operator.decide(record["id"], digest=record["prepared"]["digest"],
                                        approve=False, decided_by="operator",
                                        message="Use a smaller basis."))

    monkeypatch.setattr(journal_module, "validate", change_meanwhile)
    state = await run(graph, context, Command(resume="ok"))
    assert not (context.workspace / "energy.json").exists()
    if change == "reject":
        assert "Operator feedback: Use a smaller basis." in tool_messages(state)[-1].content
        return
    # A lapsed approval pauses for a new decision instead of failing the run.
    assert state["__interrupt__"][0].value["operation_id"] == record["id"]
    monkeypatch.undo()
    await approve(context, record)
    state = await run(graph, context, Command(resume="ok"))
    assert [a["source_path"] for a in result_artifacts(state, context.task_id)] == ["energy.json"]


@pytest.mark.asyncio
async def test_unpermitted_and_unpreparable_calls_answer_the_model(tmp_path):
    context = make_context(tmp_path)
    graph = scoped_graph([
        ai({"name": "write_file", "args": {"file_path": "/x.py", "content": "x"}, "id": "w"},
           {"name": "task", "args": {"subagent_type": "x", "description": "y"}, "id": "t"},
           run_ase_call("missing", structure="later.xyz"),
           {"name": "run_ase", "args": {"params": 1}, "id": "malformed"}),
        AIMessage(content="I cannot do that"),
    ])
    state = await run(graph, context, {"messages": [HumanMessage(content="Try")]})
    assert "__interrupt__" not in state
    messages = {m.tool_call_id: m for m in tool_messages(state)}
    assert all(m.status == "error" for m in messages.values())
    assert "not available" in messages["w"].content and "not available" in messages["t"].content
    assert "does not exist yet" in messages["missing"].content
    assert "'params' must be an object" in messages["malformed"].content
    assert state["messages"][-1].content == "I cannot do that"


@pytest.mark.asyncio
@pytest.mark.parametrize("fname, content", [
    ("empty.xyz", ""),
    ("unsupported.unknownformat", "not a structure\n"),
])
async def test_unreadable_structure_allows_a_corrected_proposal(tmp_path, fname, content):
    context = make_context(tmp_path, limits={"max_atoms": 2})
    (context.workspace / fname).write_text(content)
    graph = scoped_graph([
        ai(run_ase_call("invalid", structure=fname, output="invalid.json")),
        ai(run_ase_call("corrected", output="corrected.json")),
        AIMessage(content="Energy computed from the corrected input"),
    ])
    paused = await run(graph, context, {"messages": [HumanMessage(content="Compute energy")]})
    errors = tool_messages(paused)
    assert len(errors) == 1 and errors[0].tool_call_id == "invalid"
    assert errors[0].status == "error" and f"Cannot read input structure {fname!r}" in errors[0].content
    assert [item.value["tool_call_id"] for item in paused["__interrupt__"]] == ["corrected"]
    records = {r["tool_call_id"]: r for r in await context.journal.store.records()}
    assert records["invalid"]["state"] == "blocked" and records["invalid"]["decision"] is None
    assert records["corrected"]["state"] == "proposed"
    assert not (context.workspace / "invalid.json").exists()
    assert not (context.workspace / "corrected.json").exists()

    pending = await decide_pending(context, paused)
    state = await run(graph, context, Command(resume="ok"))
    assert state["messages"][-1].content == "Energy computed from the corrected input"
    artifacts = result_artifacts(state, context.task_id)
    assert [a["source_path"] for a in artifacts] == ["corrected.json"]
    assert json.loads(artifact_file(context.workspace, artifacts[0]).read_text())["success"] is True
    assert (await context.journal.get(pending[0]["operation_id"]))["state"] == "finalized"
    assert not (context.workspace / "invalid.json").exists()


@pytest.mark.asyncio
async def test_dependent_actions_are_reviewed_in_sequence(tmp_path):
    context = make_context(tmp_path)
    write = {"name": "smiles_to_coordinate_file",
             "args": {"smiles": "O", "output_file": "water.xyz"}, "id": "write"}
    graph = scoped_graph([
        ai(write), ai(run_ase_call("opt", structure="water.xyz", driver="opt")),
        AIMessage(content="Optimized"),
    ])
    paused = await run(graph, context, {"messages": [HumanMessage(content="Optimize water")]})
    await decide_pending(context, paused)
    paused = await run(graph, context, Command(resume="ok"))
    pending = await decide_pending(context, paused)
    assert pending[0]["prepared"]["inputs"][0]["source"] == "water.xyz"
    state = await run(graph, context, Command(resume="ok"))
    sources = sorted(a["source_path"] for a in result_artifacts(state, context.task_id))
    assert sources == ["energy.json", "water.xyz", "water_opt.traj"]


@pytest.mark.asyncio
async def test_scoped_graph_requires_context_and_async_api(tmp_path):
    graph = scoped_graph([ai(run_ase_call()), AIMessage(content="Done")])
    with pytest.raises(ScopedExecutionError, match="run_context"):
        await graph.ainvoke({"messages": [HumanMessage(content="Energy")]},
                            {"configurable": {"thread_id": "x"}})
    with pytest.raises(ValueError, match="Scoped approval"):
        normalize_approval_mode("deep_agent", "scoped")
    with pytest.raises(ValueError, match="checkpoint files"):
        construct_main_agent_graph(_ScriptedChatModel(responses=[]), scoped=True,
                                   discover_skills=False)


@pytest.mark.asyncio
async def test_chemgraph_session_runs_scoped_catalog(tmp_path, model_loader, monkeypatch):
    from chemgraph.agent.llm_agent import ChemGraph
    from chemgraph.agent.main_session import MainAgentSession

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "standalone"))
    responses, _ = model_loader
    responses.extend([ai({"name": "load_tools", "args": {"names": ["run_ase"]}, "id": "load"}),
                      ai(run_ase_call()), AIMessage(content="Energy computed")])
    context = make_context(tmp_path)
    catalog = ToolRegistry(spec for spec in ToolRegistry().specs() if spec.name in context.policy.tools)
    agent = ChemGraph(workflow_type="main_agent", approval_mode="scoped", tool_registry=catalog,
                      subagent_names=[], discover_skills=False)
    reviewed = ChemGraph(workflow_type="main_agent", approval_mode="bypass", tool_registry=catalog,
                         subagent_names=[], discover_skills=False)
    assert (agent.main_agent_metadata.graph_config.topology_fingerprint
            != reviewed.main_agent_metadata.graph_config.topology_fingerprint)
    session = MainAgentSession(agent.workflow, session_metadata=agent.main_agent_metadata)
    with run_context(context):
        turn = await session.run("Energy of H2 with EMT")
    assert turn.status == "waiting_for_user"
    await decide_pending(context, {"__interrupt__": [SimpleNamespace(value=i.payload) for i in turn.interrupts]})
    with run_context(context):
        turn = await session.resume("decided")
    assert turn.status == "completed" and turn.assistant_response == "Energy computed"
    assert [a["source_path"] for a in result_artifacts(turn.state, context.task_id)] == ["energy.json"]
    assert not (tmp_path / "standalone" / "energy.json").exists()
