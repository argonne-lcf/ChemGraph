import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from fastmcp import Client

from chemgraph.execution.base import TaskSpec
from chemgraph.execution.job_tracker import JobTracker
from chemgraph.mcp import run_mlip_mcp
from chemgraph.schemas.mlip_input import MLIPBatchInputSchema, MLIPInputSchema


class _ImmediateBackend:
    is_async_remote = False
    shares_filesystem = True

    def __init__(self):
        self.submitted = []

    def submit(self, task):
        self.submitted.append(task)
        future = Future()
        future.set_result(task.callable(**task.kwargs))
        return future


@pytest.mark.asyncio
async def test_run_mlip_mcp_exposes_standalone_tools(monkeypatch):
    backend = _ImmediateBackend()
    monkeypatch.setattr(run_mlip_mcp.mcp, "_backend", backend)
    monkeypatch.setattr(
        run_mlip_mcp,
        "run_mlip_core",
        lambda params: {
            "status": "single",
            "input": params["input_structure_file"],
        },
    )
    monkeypatch.setattr(
        run_mlip_mcp,
        "run_mlip_batch_core",
        lambda params: {
            "status": "batch",
            "total": len(params["input_structure_files"]),
        },
    )
    single = MLIPInputSchema(
        input_structure_file="input.xyz",
        model={"family": "mace", "checkpoint": "model.pt"},
    )
    batch = MLIPBatchInputSchema(
        input_structure_files=["one.xyz", "two.xyz"],
        model={"family": "mace", "checkpoint": "model.pt"},
    )

    async with Client(run_mlip_mcp.mcp) as client:
        tools = await client.list_tools()
        single_result = await client.call_tool(
            "run_mlip", {"params": single.model_dump(mode="json")}
        )
        batch_result = await client.call_tool(
            "run_mlip_batch", {"params": batch.model_dump(mode="json")}
        )

    assert {
        "run_mlip",
        "run_mlip_batch",
        "check_job_status",
        "get_job_results",
        "list_jobs",
        "cancel_job",
    } <= {tool.name for tool in tools}
    assert json.loads(single_result.content[0].text) == {
        "status": "single",
        "input": "input.xyz",
    }
    assert json.loads(batch_result.content[0].text) == {
        "status": "batch",
        "total": 2,
    }
    assert [task.callable.__name__ for task in backend.submitted] == [
        "run_mlip",
        "run_mlip_batch",
    ]


def test_mlip_remote_execution_requires_prestaged_inputs(tmp_path, monkeypatch):
    input_file = tmp_path / "input.xyz"
    input_file.write_text("local", encoding="utf-8")
    monkeypatch.setattr(
        run_mlip_mcp.mcp,
        "_backend",
        SimpleNamespace(shares_filesystem=False),
    )
    params = MLIPInputSchema(
        input_structure_file=str(input_file),
        output_results_file="/remote/results/output.json",
        model={"family": "mace", "checkpoint": "named-model"},
    )
    task = TaskSpec(
        task_id="single",
        callable=run_mlip_mcp.run_mlip,
        kwargs={"params": params.model_dump(mode="json")},
    )

    with pytest.raises(ValueError, match="Pre-stage these inputs"):
        run_mlip_mcp._mlip_transport_hook(task)


def test_mlip_remote_batch_stays_one_gpu_task(monkeypatch):
    monkeypatch.setattr(
        run_mlip_mcp.mcp,
        "_backend",
        SimpleNamespace(shares_filesystem=False),
    )
    params = MLIPBatchInputSchema(
        input_structure_files=["/remote/input/one.xyz", "/remote/input/two.xyz"],
        output_results_directory="/remote/results",
        calculator={"backend": "nvalchemi"},
        model={"family": "mace", "checkpoint": "remote-model"},
    )
    task = TaskSpec(
        task_id="batch",
        callable=run_mlip_mcp.run_mlip_batch,
        kwargs={"params": params.model_dump(mode="json")},
    )

    routed = run_mlip_mcp._mlip_transport_hook(task)

    assert routed is task
    assert routed.gpus_per_task == 1
    assert routed.callable is run_mlip_mcp.run_mlip_batch
    assert list(routed.kwargs) == ["params"]


def test_mlip_remote_execution_requires_absolute_output(monkeypatch):
    monkeypatch.setattr(
        run_mlip_mcp.mcp,
        "_backend",
        SimpleNamespace(shares_filesystem=False),
    )
    params = MLIPInputSchema(
        input_structure_file="/remote/input.xyz",
        output_results_file="output.json",
        model={"family": "mace", "checkpoint": "named-model"},
    )
    task = TaskSpec(
        task_id="single",
        callable=run_mlip_mcp.run_mlip,
        kwargs={"params": params.model_dump(mode="json")},
    )

    with pytest.raises(ValueError, match="must be an absolute path"):
        run_mlip_mcp._mlip_transport_hook(task)


@pytest.mark.asyncio
async def test_run_mlip_mcp_returns_async_job_handle(monkeypatch):
    backend = _ImmediateBackend()
    backend.is_async_remote = True
    monkeypatch.setattr(run_mlip_mcp.mcp, "_backend", backend)
    monkeypatch.setattr(run_mlip_mcp.mcp, "_tracker", JobTracker())
    monkeypatch.setattr(
        run_mlip_mcp,
        "run_mlip_core",
        lambda params: {"status": "success"},
    )
    params = MLIPInputSchema(
        input_structure_file="/remote/input.xyz",
        output_results_file="/remote/output.json",
        model={"family": "mace", "checkpoint": "named-model"},
    )

    async with Client(run_mlip_mcp.mcp) as client:
        result = await client.call_tool(
            "run_mlip", {"params": params.model_dump(mode="json")}
        )

    payload = json.loads(result.content[0].text)
    assert payload["status"] == "submitted"
    assert payload["n_tasks"] == 1
    assert payload["batch_id"]
