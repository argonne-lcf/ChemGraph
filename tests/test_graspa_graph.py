"""Exercise the original planner/executor/analyst flow without a live model."""

import asyncio
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool

from chemgraph.graphs.graspa_mcp import construct_graspa_mcp_graph
from chemgraph.mcp.data_analysis_mcp import aggregate_simulation_results, rank_mofs_performance
from chemgraph.state.graspa_state import PlannerResponse
from chemgraph.tools.graspa_analysis import write_records


def call(name, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": name}])


class Model:
    def __init__(self, root, count=1):
        self.root, self.count, self.route = root, count, "executor_subgraph"
        self.plans, self.executor_contexts, self.reports = [], [], []
        self.paths = [str(root / f"task-{index}.jsonl") for index in range(count)]

    def with_structured_output(self, schema):
        assert schema is PlannerResponse
        owner = self

        class Planner:
            def invoke(self, messages):
                owner.plans.append(str(messages))
                step = "insight_analyst" if "UPDATED: Results" in str(messages) else owner.route
                return schema(thought_process="Reviewing the request.", next_step=step,
                              tasks=[{"task_index": i + 1, "prompt": f"ensemble {i}"}
                                     for i in range(owner.count)] if step == "executor_subgraph" else [])
        return Planner()

    def bind_tools(self, tools):
        owner = self
        executor = any(t.name == "run_graspa_ensemble" for t in tools)

        class Bound:
            async def ainvoke(self, messages):
                assert executor
                owner.executor_contexts.append(str(messages))
                last = messages[-1]
                if not isinstance(last, ToolMessage):
                    return call("run_graspa_ensemble", index=int(last.content.rsplit(" ", 1)[-1]))
                output = json.loads(last.content)
                if last.name == "run_graspa_ensemble" or (
                    last.name == "check_job_status" and output["status"] == "pending"
                ):
                    return call("check_job_status", batch_id=output["batch_id"])
                if last.name == "check_job_status":
                    return call("get_job_results", batch_id=output["batch_id"])
                return AIMessage(content=f"{output['status']}: {output['records_path']}")

            def invoke(self, messages):
                assert not executor
                owner.reports.append(str(messages))
                if len(owner.reports) == 1:
                    return call("aggregate_simulation_results", file_paths=owner.paths,
                                output_csv_path=str(owner.root / "results.csv"))
                if len(owner.reports) == 2:
                    return call("rank_mofs_performance", input_csv_path=str(owner.root / "results.csv"),
                                ads_pressure=960, ads_temp=298, des_pressure=320, des_temp=298,
                                min_cutoff=5)
                return AIMessage(content="Reported the ranking and failed outcomes.")
        return Bound()


def setup(tmp_path, count=1, failed=False, blocked=False):
    model = Model(tmp_path, count)
    calls, polls = [], {}
    waiting, release = asyncio.Event(), asyncio.Event()

    @tool
    async def run_graspa_ensemble(index: int) -> dict:
        """Submit an ensemble."""
        calls.append(index)
        return {"status": "submitted", "batch_id": str(index)}

    @tool
    async def check_job_status(batch_id: str) -> dict:
        """Check an ensemble."""
        if blocked and batch_id == "0":
            waiting.set()
            await release.wait()
        polls[batch_id] = polls.get(batch_id, 0) + 1
        return {"status": "pending" if polls[batch_id] == 1 else "completed", "batch_id": batch_id}

    @tool
    async def get_job_results(batch_id: str) -> dict:
        """Collect terminal records."""
        index = int(batch_id)
        write_records(tmp_path / f"task-{index}.jsonl", [
            {"input_structure_file": f"/shared/{index}.cif", "temperature_in_K": 298,
             "pressure_in_Pa": p, "uptake_in_mol_kg": p / 100,
             "status": "failure" if failed and index == 0 else "success"} for p in (960, 320)
        ])
        return {"status": "failed" if failed and index == 0 else "completed",
                "records_path": model.paths[index]}

    workflow = construct_graspa_mcp_graph(
        model, executor_tools=[run_graspa_ensemble, check_job_status, get_job_results],
        analysis_tools=[StructuredTool.from_function(fn) for fn in
                        (aggregate_simulation_results, rank_mofs_performance)], checkpointer=None,
    )
    return model, workflow, calls, polls, waiting, release


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_executor_tool_loops_and_analyst_ranking(tmp_path, monkeypatch, failed):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    model, graph, calls, polls, _, _ = setup(tmp_path, count=2, failed=failed)
    query = "Screen /shared/cifs at 298 K, 960/320 Pa, 12345 cycles; output /shared/out; min_cutoff=5."
    result = await graph.ainvoke({"messages": query}, {"recursion_limit": 30})
    assert sorted(calls) == [0, 1]
    assert polls == {"0": 2, "1": 2}
    assert len(result["executor_results"]) == 2
    assert all(isinstance(item, str) for item in result["executor_results"])
    assert set(result["executor_logs"]) == {"worker_1", "worker_2"}
    assert len(model.plans) == 2
    assert len(model.reports) == 3
    assert all(query in context for context in model.executor_contexts + model.reports)
    names = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    assert names == ["aggregate_simulation_results", "rank_mofs_performance"]
    ranking = [m.content for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "rank_mofs_performance"][0]
    assert "Values >= 5.0 mol/kg" in ranking
    assert f"excluded {int(failed)} incomplete/failed structures" in ranking
    assert (tmp_path / "results.csv").exists()
    assert any(("failed" if failed else "completed") in item for item in result["executor_results"])


@pytest.mark.asyncio
async def test_fanout_joins_before_replanning_or_analysis(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    model, graph, calls, _, waiting, release = setup(tmp_path, count=2, blocked=True)
    running = asyncio.create_task(graph.ainvoke({"messages": "screen"}, {"recursion_limit": 30}))
    try:
        await asyncio.wait_for(waiting.wait(), 5)
        assert not model.reports
        assert len(model.plans) == 1
        release.set()
        result = await asyncio.wait_for(running, 10)
        assert len(result["executor_results"]) == 2
        assert sorted(calls) == [0, 1]
    finally:
        if not running.done():
            running.cancel()
        await asyncio.gather(running, return_exceptions=True)


@pytest.mark.asyncio
async def test_followup_can_finish_without_submission_and_preserves_history(tmp_path):
    model, graph, calls, _, _, _ = setup(tmp_path)
    model.route = "FINISH"
    await graph.ainvoke({"messages": [HumanMessage(content="Previous screening at 298 K."),
                                      AIMessage(content="Results are saved."),
                                      HumanMessage(content="Explain the previous result; do not run again.")]})
    assert not calls
    assert "Previous screening at 298 K" in model.plans[0]
    assert "do not run again" in model.plans[0]


@pytest.mark.asyncio
async def test_analysis_only_uses_existing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    model, graph, calls, _, _, _ = setup(tmp_path)
    model.route = "insight_analyst"
    write_records(tmp_path / "task-0.jsonl", [
        {"input_structure_file": "/shared/0.cif", "temperature_in_K": 298, "pressure_in_Pa": p,
         "uptake_in_mol_kg": p / 100, "status": "success"} for p in (960, 320)
    ])
    await graph.ainvoke({"messages": "Analyze saved results with min_cutoff=5; do not simulate."})
    assert not calls
    assert len(model.reports) == 3


def test_public_constructor_preserves_prompt_and_tool_overrides(tmp_path, monkeypatch):
    from chemgraph.agent.llm_agent import ChemGraph, PromptConfig
    from tests.test_graphs import _fake_prepared

    monkeypatch.setattr("chemgraph.agent.llm_agent.load_chat_model_prepared", _fake_prepared)
    captured = {}
    monkeypatch.setattr("chemgraph.agent.llm_agent.construct_graspa_mcp_graph", lambda **kw: captured.update(kw))
    tools, data_tools = [object()], [object()]
    ChemGraph(workflow_type="graspa_mcp", tools=tools, data_tools=data_tools, enable_memory=False,
              log_dir=str(tmp_path), prompts=PromptConfig(planner="plan", executor="execute", aggregator="analyze"))
    assert captured["planner_prompt"] == "plan"
    assert captured["executor_prompt"] == "execute"
    assert captured["analyst_prompt"] == "analyze"
    assert captured["executor_tools"] == tools
    assert captured["analysis_tools"] == data_tools
    assert "options" not in captured
