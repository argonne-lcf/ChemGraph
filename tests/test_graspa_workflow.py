"""Exercise the real LangChain MCP adapter with hermetic backend responses."""
import asyncio
from collections import Counter
import json

import pytest
from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
from mcp.types import CallToolResult, TextContent, Tool

from chemgraph.execution.graspa_workflow import (
    GraspaCollector, load_journal, run_lock, save_journal,
)
from chemgraph.schemas.graspa_workflow import GraspaWorkflowOptions
from chemgraph.tools.graspa_analysis import write_records


class Backend:
    def __init__(self, root, *, mode="async"):
        self.root, self.mode = root, mode
        self.calls = Counter()
        self.batches = {}
        self.artifact = True
        self.wrapped = False
        self.lost_reply = False
        self.block = False
        self.waiting = asyncio.Event()
        self.release = asyncio.Event()
        self.mutate = lambda rows: rows
        self.failed_sources = set()
        self.fail_all = False

    def rows(self, batch_id, params):
        paths = params.get("input_structures")
        if not paths:
            paths = ["/remote/a.CIF"]
        return [
            {"job_id": f"{batch_id}-{i}", "input_structure_file": source,
             "temperature_in_K": condition["temperature"], "pressure_in_Pa": condition["pressure"],
             "status": "failure" if self.fail_all or source in self.failed_sources else "success",
             "uptake_in_mol_kg": None if self.fail_all or source in self.failed_sources
             else condition["pressure"] / 100,
             "adsorbate": "H2O"}
            for i, (source, condition) in enumerate(
                (source, condition) for source in paths for condition in params["conditions"])
        ]

    async def call_tool(self, name, arguments=None, **_kwargs):
        self.calls[name] += 1
        if name == "run_graspa_ensemble":
            batch_id = f"batch-{self.calls[name]}"
            rows = self.rows(batch_id, arguments["params"])
            self.batches[batch_id] = rows
            if self.lost_reply:
                raise ConnectionError("response lost after acceptance")
            if self.mode == "legacy":
                path = self.root / "legacy 'summary.jsonl"
                write_records(path, rows)
                result = f"Ensemble execution completed. Ran {len(rows)} tasks ({len(rows)} successful). Detailed results saved to '{path}'."
            elif self.mode == "immediate":
                result = {"status": "completed", "results": self.mutate(rows)}
            else:
                result = {"status": "submitted", "batch_id": batch_id, "n_tasks": len(rows)}
        elif name == "check_job_status":
            if self.block:
                self.waiting.set()
                await self.release.wait()
            rows = self.batches[arguments["batch_id"]]
            result = {"status": "completed", "total_tasks": len(rows), "pending_tasks": 0}
        else:
            result = {"results": self.mutate(self.batches[arguments["batch_id"]])}
        if self.wrapped:
            result = {"result": result}
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(result))],
            structuredContent=result if self.artifact and isinstance(result, dict) else None,
        )

    def tools(self):
        return [convert_mcp_tool_to_langchain_tool(self, Tool(
            name=name, description=name,
            inputSchema={"type": "object", "properties": properties},
        )) for name, properties in (
            ("run_graspa_ensemble", {"params": {"type": "object"}}),
            ("check_job_status", {"batch_id": {"type": "string"}}),
            ("get_job_results", {"batch_id": {"type": "string"}, "include_partial": {"type": "boolean"}}),
        )]


def request(paths=None):
    return {"input_structures": paths or ["/one/same.cif", "/two/same.cif"], "adsorbate": "H2O",
            "conditions": [{"temperature": 298.0, "pressure": 960.0}, {"temperature": 298.0, "pressure": 320.0}]}


def initialize(root, params=None):
    root.mkdir(exist_ok=True)
    save_journal(root, {"requests": {"task_1": params or request()}, "batches": {}})


def collector(root, backend, **options):
    return GraspaCollector(root, backend.tools(), GraspaWorkflowOptions(
        poll_interval_seconds=0.001, wait_timeout_seconds=options.pop("wait_timeout_seconds", 2), **options,
    ))


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,artifact,wrapped", [
    ("async", True, False), ("immediate", True, False), ("immediate", False, False),
    ("legacy", False, False), ("legacy", True, True),
])
async def test_real_adapter_collects_and_resumes_without_resubmitting(tmp_path, mode, artifact, wrapped):
    initialize(tmp_path)
    backend = Backend(tmp_path, mode=mode)
    backend.artifact, backend.wrapped = artifact, wrapped
    first = await collector(tmp_path, backend).collect_all()
    assert first["task_1"]["status"] == "completed"
    assert first["task_1"]["total_records"] == 4
    assert await collector(tmp_path, backend).collect_all() == first
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.asyncio
async def test_timeout_retains_id_and_resume_collects_original_batch(tmp_path, monkeypatch):
    initialize(tmp_path)
    backend = Backend(tmp_path)
    backend.block = True
    original_timeout = asyncio.timeout
    deadline = original_timeout(None)
    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "timeout", lambda _seconds: deadline)
        task = asyncio.create_task(collector(tmp_path, backend).collect_all())
        await asyncio.wait_for(backend.waiting.wait(), 5)
        deadline.reschedule(asyncio.get_running_loop().time())
        first = await task
    assert first["task_1"]["status"] == "collection_error"
    assert first["task_1"]["batch_id"] == "batch-1"
    backend.release.set()
    assert (await collector(tmp_path, backend).collect_all())["task_1"]["status"] == "completed"
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.asyncio
async def test_cancel_propagates_and_releases_run_lock(tmp_path):
    initialize(tmp_path)
    backend = Backend(tmp_path)
    backend.block = True
    task = asyncio.create_task(collector(tmp_path, backend).collect_all())
    await asyncio.wait_for(backend.waiting.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with run_lock(tmp_path):
        entry = load_journal(tmp_path)["batches"]["task_1"]
        assert entry["phase"] == "submitted"
        assert entry["batch_id"] == "batch-1"


@pytest.mark.asyncio
async def test_unknown_submission_is_never_automatically_retried(tmp_path):
    initialize(tmp_path)
    backend = Backend(tmp_path)
    backend.lost_reply = True
    for _ in range(2):
        result = await collector(tmp_path, backend).collect_all()
        assert result["task_1"]["status"] == "collection_error"
    assert backend.calls["run_graspa_ensemble"] == 1
    assert load_journal(tmp_path)["batches"]["task_1"]["phase"] == "submission_unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "condition", "source"])
async def test_bad_records_cannot_complete_a_batch(tmp_path, corruption):
    initialize(tmp_path)
    backend = Backend(tmp_path)

    def mutate(rows):
        if corruption == "missing":
            return rows[:-1]
        rows = [dict(row) for row in rows]
        key = {"duplicate": "job_id", "condition": "pressure_in_Pa", "source": "input_structure_file"}[corruption]
        rows[1][key] = rows[0]["job_id"] if corruption == "duplicate" else -5 if corruption == "condition" else "/unexpected.cif"
        return rows

    backend.mutate = mutate
    result = await collector(tmp_path, backend).collect_all()
    assert result["task_1"]["status"] == "collection_error"
    assert result["task_1"]["batch_id"] == "batch-1"


@pytest.mark.asyncio
async def test_remote_inputs_are_not_resolved_on_the_client(tmp_path):
    params = request()
    params.update(input_structures="", remote_structure_directory="/worker/staged")
    initialize(tmp_path, params)
    backend = Backend(tmp_path)
    assert (await collector(tmp_path, backend).collect_all())["task_1"]["status"] == "completed"


def test_concurrent_clients_cannot_write_the_same_run(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another client"):
            with run_lock(tmp_path):
                pass


@pytest.mark.asyncio
async def test_pending_running_partial_collection(tmp_path):
    initialize(tmp_path)
    backend = Backend(tmp_path)
    original = backend.call_tool
    statuses = iter(["pending", "running", "partial"])
    finished = 0

    async def call(name, arguments=None, **kwargs):
        nonlocal finished
        if name == "check_job_status":
            state = next(statuses)
            finished = {"pending": 0, "running": 1, "partial": 4}[state]
            backend.calls[name] += 1
            return CallToolResult(content=[], structuredContent={"status": state, "total_tasks": 4})
        if name == "get_job_results":
            assert arguments["include_partial"]
            backend.mutate = lambda rows: rows[:finished]
        return await original(name, arguments, **kwargs)

    backend.call_tool = call
    backend.failed_sources.add("/two/same.cif")
    result = (await collector(tmp_path, backend).collect_all())["task_1"]
    assert result["status"] == "partial"
    assert result["failed_records"] == 2
    assert backend.calls["check_job_status"] == 3


@pytest.mark.asyncio
async def test_unreadable_legacy_file_is_not_resubmitted(tmp_path):
    initialize(tmp_path)
    backend = Backend(tmp_path, mode="legacy")
    original = backend.call_tool

    async def call(*args, **kwargs):
        response = await original(*args, **kwargs)
        (tmp_path / "legacy 'summary.jsonl").unlink()
        return response

    backend.call_tool = call
    for _ in range(2):
        result = (await collector(tmp_path, backend).collect_all())["task_1"]
        assert result["status"] == "collection_error"
        assert "readable on the client" in result["message"]
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.asyncio
async def test_cancel_before_acknowledgment_requires_reconciliation(tmp_path):
    initialize(tmp_path)
    backend = Backend(tmp_path)
    original = backend.call_tool

    async def call(*args, **kwargs):
        response = await original(*args, **kwargs)
        backend.waiting.set()
        await backend.release.wait()
        return response

    backend.call_tool = call
    task = asyncio.create_task(collector(tmp_path, backend).collect_all())
    await asyncio.wait_for(backend.waiting.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert load_journal(tmp_path)["batches"]["task_1"]["phase"] == "submission_unknown"
    assert (await collector(tmp_path, backend).collect_all())["task_1"]["status"] == "collection_error"
    assert backend.calls["run_graspa_ensemble"] == 1


@pytest.mark.parametrize("options", [
    {"resume": True}, {"run_directory": ""}, {"poll_interval_seconds": 0},
    {"wait_timeout_seconds": -1}, {"poll_interval_seconds": float("nan")},
    {"wait_timeout_seconds": float("inf")}, {"unknown_option": True},
])
def test_invalid_workflow_options(options):
    with pytest.raises(ValueError):
        GraspaWorkflowOptions.model_validate(options)
