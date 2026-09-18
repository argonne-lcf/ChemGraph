import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from chemgraph.graphs.graspa_mcp import construct_graspa_mcp_graph
from chemgraph.schemas.graspa_workflow import GraspaPlan
from tests.test_graspa_workflow import Backend


class Model:
    def __init__(self, directory):
        self.directory = directory
        self.preparations = 0
        self.reports = []
        self.plan = {"tasks": [{"task_index": 1, "prompt": str(directory)}],
                     "analysis": {"adsorption": {"temperature": 298, "pressure": 960},
                                  "desorption": {"temperature": 298, "pressure": 320}, "top_fraction": 0.2}}

    def with_structured_output(self, schema):
        owner = self

        class Structured:
            async def ainvoke(self, messages, config=None):
                owner.preparations += 1
                if schema is GraspaPlan:
                    return schema.model_validate(owner.plan)
                return schema(
                    input_structures=str(owner.directory), adsorbate="H2O", n_cycles=100,
                    conditions=[{"temperature": 298, "pressure": 960}, {"temperature": 298, "pressure": 320}],
                )
        return Structured()

    async def ainvoke(self, messages, config=None):
        self.reports.append(messages[-1].content)
        return AIMessage(content="See the canonical analysis artifacts.")


def setup(tmp_path, count=2, **backend_args):
    cifs = tmp_path / "cifs"
    cifs.mkdir()
    for i in range(count):
        (cifs / f"{i}.CIF").touch()
    return Model(cifs), Backend(tmp_path, **backend_args), tmp_path / "run"


def graph(model, backend, root, **options):
    return construct_graspa_mcp_graph(
        model, executor_tools=backend.tools(), checkpointer=None,
        options={"run_directory": str(root), "poll_interval_seconds": 0.001,
                 "wait_timeout_seconds": 3, **options},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["async", "immediate", "legacy"])
async def test_native_graph_and_resume_bypass_preparation(tmp_path, mode):
    model, backend, root = setup(tmp_path, mode=mode)
    result = await graph(model, backend, root).ainvoke({"messages": "screen"}, {"recursion_limit": 8})
    assert result["workflow_status"] == "completed"
    assert result["analysis"]["valid_structures"] == 2
    assert result["analysis"]["preview"][0]["working_capacity"] == pytest.approx(6.4)
    assert model.preparations == 2
    assert backend.calls["run_graspa_ensemble"] == 1
    assert "requests" not in result
    assert "results" not in result["executor_results"]["task_1"]
    resumed = await graph(model, backend, root, resume=True).ainvoke({"messages": "screen"})
    assert resumed["analysis"] == result["analysis"]
    assert model.preparations == 2
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.asyncio
async def test_analysis_waits_for_all_executors_and_keeps_failures(tmp_path):
    model, backend, root = setup(tmp_path)
    model.plan["tasks"].append({"task_index": 2, "prompt": str(model.directory)})
    backend.failed_sources.add(str(model.directory / "1.CIF"))
    backend.block = True
    task = asyncio.create_task(graph(model, backend, root).ainvoke({"messages": "screen"}))
    await asyncio.wait_for(backend.waiting.wait(), 2)
    assert not (root / "analysis.json").exists()
    assert not model.reports
    backend.release.set()
    result = await task
    assert len(result["executor_results"]) == 2
    assert result["workflow_status"] == "partial"
    assert result["analysis"]["excluded_structures"] == 1


@pytest.mark.asyncio
async def test_incomplete_collection_never_produces_rankings(tmp_path):
    model, backend, root = setup(tmp_path)
    backend.mutate = lambda rows: rows[:-1]
    result = await graph(model, backend, root).ainvoke({"messages": "screen"})
    assert result["workflow_status"] == "incomplete"
    assert not (root / "rankings.csv").exists()


@pytest.mark.asyncio
async def test_large_ensemble_keeps_model_and_checkpoint_payloads_bounded(tmp_path, monkeypatch):
    model, backend, root = setup(tmp_path)
    # Freeze an explicit workload via the preparation seam without involving
    # a model in enumeration or executing any chemistry.
    from chemgraph.mcp import graspa_mcp_hpc
    monkeypatch.setattr(graspa_mcp_hpc, "_local_structure_files", lambda _source: [f"/shared/{i}.cif" for i in range(4608)])
    result = await graph(model, backend, root).ainvoke({"messages": "screen"})
    assert result["analysis"]["total_records"] == 9216
    assert result["analysis"]["selected_structures"] == 922
    assert len(model.reports[-1]) < 5000
    assert len(json.dumps({k: v for k, v in result.items() if k != "messages"})) < 8000


@pytest.mark.asyncio
async def test_empty_plan_fails_before_submission(tmp_path):
    model, backend, root = setup(tmp_path)
    model.plan["tasks"] = []
    with pytest.raises(ValueError):
        await graph(model, backend, root).ainvoke({"messages": "screen"})
    assert not backend.calls


@pytest.mark.asyncio
async def test_existing_directory_and_changed_resume_query_are_rejected(tmp_path):
    model, backend, root = setup(tmp_path)
    await graph(model, backend, root).ainvoke({"messages": "screen"})
    with pytest.raises(ValueError, match="resume=True"):
        await graph(model, backend, root).ainvoke({"messages": "screen"})
    with pytest.raises(ValueError, match="original query"):
        await graph(model, backend, root, resume=True).ainvoke({"messages": "different"})
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_preparation_concurrency_and_sibling_cleanup(tmp_path, fail):
    model, backend, root = setup(tmp_path)
    model.plan["tasks"] = [{"task_index": i + 1, "prompt": str(i)} for i in range(5)]
    original = model.with_structured_output
    active = peak = finished = 0
    two_started = asyncio.Event()

    def structured(schema):
        delegate = original(schema)
        if schema is GraspaPlan:
            return delegate

        class Preparation:
            async def ainvoke(self, messages, config=None):
                nonlocal active, peak, finished
                active += 1
                peak = max(peak, active)
                if active == 2:
                    two_started.set()
                try:
                    await two_started.wait()
                    if fail:
                        if messages[-1].content == "0":
                            raise ValueError("bad preparation")
                        await asyncio.Event().wait()
                    return await delegate.ainvoke(messages, config)
                finally:
                    active -= 1
                    finished += 1
        return Preparation()

    model.with_structured_output = structured
    operation = graph(model, backend, root).ainvoke({"messages": "screen"}, {"max_concurrency": 2})
    if fail:
        with pytest.raises(ValueError, match="bad preparation"):
            await asyncio.wait_for(operation, 5)
        assert not backend.calls
        assert not json.loads((root / "workflow.json").read_text())["requests"]
    else:
        await asyncio.wait_for(operation, 5)
        assert finished == 5
    assert peak == 2
    assert active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("log_dir", [None, "logs"])
async def test_relative_log_directory_is_resolved_once(tmp_path, monkeypatch, log_dir):
    model, backend, _ = setup(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", "logs")
    workflow = construct_graspa_mcp_graph(model, executor_tools=backend.tools(), checkpointer=None, log_dir=log_dir)
    state = await workflow.ainvoke({"messages": "screen"})
    assert state["run_directory"].startswith(str(tmp_path / "logs/graspa_workflows") + "/")


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_plan", ["duplicates", "conditions"])
async def test_invalid_plan_or_condition_never_submits(tmp_path, bad_plan):
    model, backend, root = setup(tmp_path)
    if bad_plan == "duplicates":
        model.plan["tasks"] *= 2
    else:
        model.plan["analysis"]["adsorption"]["temperature"] = 298.15
    with pytest.raises(ValueError):
        await graph(model, backend, root).ainvoke({"messages": "screen"})
    assert not backend.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("return_option", ["state", "last_message"])
@pytest.mark.parametrize("status", ["completed", "partial", "failed", "incomplete"])
async def test_public_chemgraph_options_return_value_and_event(tmp_path, monkeypatch, return_option, status):
    from chemgraph.agent.llm_agent import ChemGraph
    from tests.test_graphs import _fake_prepared

    model, backend, root = setup(tmp_path)
    if status == "partial":
        backend.failed_sources.add(str(model.directory / "1.CIF"))
    elif status == "failed":
        backend.fail_all = True
    elif status == "incomplete":
        backend.mutate = lambda rows: rows[:-1]
    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", lambda **kw: (model, _fake_prepared()[1]))
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path / "logs"))
    events = []
    agent = ChemGraph(workflow_type="graspa_mcp", tools=backend.tools(), enable_memory=False,
                      log_dir=str(tmp_path / "logs"), return_option=return_option,
                      graspa_options={"run_directory": str(root)},
                      on_event=lambda name, payload: events.append((name, payload)))
    result = await agent.run("screen")
    if return_option == "state":
        assert result["workflow_status"] == status
        assert result["run_directory"] == str(root)
    else:
        assert isinstance(result, AIMessage)
    assert [payload["status"] for name, payload in events if name == "workflow_finished"] == [status]
    assert json.loads((root / "analysis.json").read_text())["status"] == status


def test_public_options_and_prompt_overrides(tmp_path, monkeypatch):
    from chemgraph.agent.llm_agent import ChemGraph, PromptConfig
    from tests.test_graphs import _fake_prepared

    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", _fake_prepared)
    captured = {}
    monkeypatch.setattr("chemgraph.agent.llm_agent.construct_graspa_mcp_graph", lambda **kw: captured.update(kw))
    ChemGraph(workflow_type="graspa_mcp", tools=[object()], enable_memory=False, log_dir=str(tmp_path),
              prompts=PromptConfig(planner="plan", executor="prepare", aggregator="explain"))
    assert captured["planner_prompt"] == "plan"
    assert captured["executor_prompt"] == "prepare"
    assert captured["analyst_prompt"] == "explain"
    with pytest.raises(ValueError, match="requires workflow_type"):
        ChemGraph(graspa_options={})


@pytest.mark.asyncio
async def test_corrupt_saved_records_remove_stale_rankings_on_resume(tmp_path):
    model, backend, root = setup(tmp_path)
    await graph(model, backend, root).ainvoke({"messages": "screen"})
    assert (root / "rankings.csv").exists()
    (root / "task_1.jsonl").write_text("corrupted JSONL\n")
    result = await graph(model, backend, root, resume=True).ainvoke({"messages": "screen"})
    assert result["workflow_status"] == "incomplete"
    assert result["analysis"]["conditions"]["adsorption"]["pressure"] == 960
    assert not (root / "rankings.csv").exists()
    assert not (root / "top_candidates.csv").exists()
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.asyncio
async def test_report_error_preserves_scientific_outcome(tmp_path):
    model, backend, root = setup(tmp_path)

    async def unavailable(*args, **kwargs):
        raise RuntimeError("model unavailable")

    model.ainvoke = unavailable
    result = await graph(model, backend, root).ainvoke({"messages": "screen"})
    assert result["workflow_status"] == "completed"
    assert (root / "top_candidates.csv").exists()
    assert json.loads((root / "report_error.json").read_text())["message"] == "model unavailable"
    assert "Canonical analysis" in result["messages"][-1].content
