"""Exercise gRASPA through registered MCP tools without HPC services."""

import asyncio
from concurrent.futures import Future
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from ase import Atoms
from ase.io import write
import pytest

pytest.importorskip("fastmcp")
from fastmcp import Client
from fastmcp.exceptions import ToolError

from chemgraph.execution.job_tracker import JobTracker
from chemgraph.mcp import graspa_mcp_hpc as hpc
from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools import graspa_core


def resolved(value=None, error=None):
    future = Future()
    if error is None:
        future.set_result(value)
    else:
        future.set_exception(error)
    return future


class Backend:
    shares_filesystem = True
    is_async_remote = False

    def __init__(self):
        self.tasks = []
        self.on_submit = None

    def submit(self, task):
        self.tasks.append(task)
        if self.on_submit is not None:
            return self.on_submit(task, len(self.tasks))
        try:
            return resolved(task.callable(**task.kwargs))
        except Exception as exc:
            return resolved(error=exc)


@pytest.fixture
def ensemble(tmp_path, monkeypatch):
    source = tmp_path / "MOF.CIF"
    write(source, Atoms("C", cell=[30, 30, 30], pbc=True), format="cif")
    server = hpc.mcp
    if server._backend_kwargs is None:
        server.init_backend()
    backend = Backend()
    monkeypatch.setattr(server, "_backend", backend)
    monkeypatch.setattr(server, "_tracker", JobTracker(persist_file=tmp_path / "jobs.json"))
    real_core = graspa_core.run_graspa_core
    core = Mock(return_value={"status": "success", "uptake_in_mol_kg": 1.25})
    monkeypatch.setattr(graspa_core, "run_graspa_core", core)
    return SimpleNamespace(
        server=server, backend=backend, source=source, core=core, real_core=real_core,
        params={"input_structures": [str(source)], "adsorbate": "H2O"},
    )


async def run(ensemble, **updates):
    async with Client(ensemble.server) as client:
        result = await client.call_tool(
            "run_graspa_ensemble", {"params": {**ensemble.params, **updates}},
        )
        return result.structured_content


def job_response(response):
    # Existing job tools retain the SDK's text-only dict return schema.
    return json.loads(response.content[0].text)


@pytest.mark.asyncio
async def test_registered_schema_and_single_item(ensemble):
    async with Client(ensemble.server) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
    assert {"check_job_status", "get_job_results", "list_jobs", "cancel_job"} <= tools.keys()
    tool = tools["run_graspa_ensemble"]
    assert tool.inputSchema["required"] == ["params"]
    schema = tool.inputSchema["$defs"]["graspa_input_schema_ensemble"]
    assert schema["properties"]["timeout_seconds"]["default"] is None
    assert schema["properties"]["discovery_timeout_seconds"]["default"] == 30
    assert tool.outputSchema["type"] == "object"
    result = await run(ensemble)
    assert result["status"] == "completed"
    record, = result["results"]
    assert record["status"] == "success"
    assert record["uptake_in_mol_kg"] == 1.25
    assert record["job_id"] == record["task_id"] == ensemble.backend.tasks[0].task_id
    assert record["input_structure_file"] == str(ensemble.source)
    assert record["temperature"] == record["temperature_in_K"] == 298.15


@pytest.mark.asyncio
async def test_duplicates_conditions_and_directory_filtering(ensemble, tmp_path):
    conditions = [{"temperature": 300, "pressure": 10}, {"temperature": 310, "pressure": 20}]
    result = await run(
        ensemble, input_structures=[str(ensemble.source)] * 2, conditions=conditions,
    )
    records = result["results"]
    assert len({r["job_id"] for r in records}) == 4
    assert [(r["temperature"], r["pressure"]) for r in records] == [(300, 10), (310, 20)] * 2
    (tmp_path / "ignore.xyz").touch()
    (tmp_path / "directory.cif").mkdir()
    assert len((await run(ensemble, input_structures=str(tmp_path)))["results"]) == 1


@pytest.mark.asyncio
async def test_listed_filename_resolves_under_log_dir(ensemble, monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(ensemble.source.parent))
    record, = (await run(ensemble, input_structures=[ensemble.source.name]))["results"]
    assert record["input_structure_file"] == str(ensemble.source)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["empty", "missing", "wrong_extension", "directory", "both", "neither", "no_conditions"])
async def test_invalid_inputs_submit_nothing(ensemble, tmp_path, kind):
    updates = {}
    empty = tmp_path / "empty"
    empty.mkdir()
    if kind == "empty":
        updates["input_structures"] = str(empty)
    elif kind == "missing":
        updates["input_structures"] = [str(ensemble.source), str(tmp_path / "missing.cif")]
    elif kind == "wrong_extension":
        other = tmp_path / "other.xyz"
        other.touch()
        updates["input_structures"] = [str(other)]
    elif kind == "directory":
        updates["input_structures"] = [str(empty)]
    elif kind == "both":
        updates["remote_structure_directory"] = "/remote/cifs"
    elif kind == "neither":
        updates["input_structures"] = []
    else:
        updates["conditions"] = []
    with pytest.raises(ToolError):
        await run(ensemble, **updates)
    assert not ensemble.backend.tasks


@pytest.mark.asyncio
async def test_nonshared_local_mode_rejected(ensemble):
    ensemble.backend.shares_filesystem = False
    with pytest.raises(ToolError, match="pre-stage CIFs.*remote_structure_directory"):
        await run(ensemble)
    assert not ensemble.backend.tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("path", [
    "/worker/staged/My MOF.CIF",
    r"C:\staged\My MOF.CIF",
    r"\\server\share\My MOF.CIF",
])
async def test_remote_paths_use_worker_syntax_and_preserve_source(ensemble, path):
    ensemble.backend.shares_filesystem = False

    def submit(task, count):
        if count == 1:
            return resolved([path])
        return resolved(task.callable(**task.kwargs))

    ensemble.backend.on_submit = submit
    record, = (await run(
        ensemble, input_structures="", remote_structure_directory="~/staged",
    ))["results"]
    assert record["status"] == "success"
    assert record["structure"] == "My MOF"
    assert record["input_structure_file"] == path
    assert ensemble.core.call_args.args[0].input_structure_file == path


@pytest.mark.asyncio
@pytest.mark.parametrize("paths", [
    "not a list", [None], [12], ["relative.cif"], ["~/MOF.cif"],
    [r"C:relative.cif"], [r"\staged\MOF.cif"],
    ["/worker/MOF.xyz"], [r"C:\staged\MOF.xyz"],
])
async def test_invalid_remote_paths_submit_no_simulations(ensemble, paths):
    ensemble.backend.on_submit = lambda task, count: resolved(paths)
    with pytest.raises(ToolError, match="Discovery must return absolute CIF paths"):
        await run(
            ensemble, input_structures="", remote_structure_directory="~/staged",
        )
    assert len(ensemble.backend.tasks) == 1
    ensemble.core.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [30, None])
async def test_remote_discovery_is_awaited_and_responsive(ensemble, timeout):
    ensemble.backend.shares_filesystem = False
    discovery = Future()
    submitted = asyncio.Event()

    def submit(task, count):
        if count == 1:
            submitted.set()
            assert task.kwargs == {"path": "~/staged"}
            return discovery
        return resolved(task.callable(**task.kwargs))

    ensemble.backend.on_submit = submit
    request = asyncio.create_task(run(
        ensemble, input_structures="", remote_structure_directory="~/staged",
        discovery_timeout_seconds=timeout, output_directory="worker-output",
        timeout_seconds=7,
    ))
    await asyncio.wait_for(submitted.wait(), 2)
    # This coroutine can advance while the discovery future is unresolved.
    await asyncio.sleep(0)
    assert not request.done()
    discovery.set_result(["/worker/staged/MOF.CIF"])
    result = await asyncio.wait_for(request, 2)
    record, = result["results"]
    assert record["input_structure_file"] == "/worker/staged/MOF.CIF"
    worker_input = ensemble.core.call_args.args[0]
    assert worker_input.output_directory == "worker-output"
    assert worker_input.output_result_file == "raspa.log"
    assert worker_input.timeout_seconds == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["submit", "worker", "timeout", "empty", "malformed"])
async def test_discovery_failure_is_actionable_and_does_not_fan_out(ensemble, failure):
    def submit(task, count):
        if failure == "submit":
            raise RuntimeError("endpoint rejected probe")
        if failure == "worker":
            return resolved(error=FileNotFoundError("staged directory missing"))
        if failure == "timeout":
            return Future()
        return resolved([] if failure == "empty" else ["relative.cif"])

    ensemble.backend.on_submit = submit
    with pytest.raises(ToolError, match="Could not discover CIFs.*No simulations were submitted"):
        await run(
            ensemble, input_structures="", remote_structure_directory="/worker/cifs",
            discovery_timeout_seconds=0.02,
        )
    assert len(ensemble.backend.tasks) == 1
    ensemble.core.assert_not_called()


def test_discovery_resolves_paths_on_worker(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "staged"
    directory.mkdir()
    (directory / "MOF.CIF").touch()
    (directory / "directory.cif").mkdir()
    (directory / "ignore.xyz").touch()
    assert hpc._ls_remote_files("staged") == [str(directory / "MOF.CIF")]


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    1.25, {}, {"uptake_in_mol_kg": 1.25}, {"status": "pending"},
    *[{"status": "success", "uptake_in_mol_kg": value} for value in (None, True, "1", float("nan"), float("inf"), -1)],
])
async def test_malformed_core_results_are_failures(ensemble, raw):
    ensemble.core.return_value = raw
    record, = (await run(ensemble))["results"]
    assert record["status"] == "failure"
    assert record["uptake_in_mol_kg"] is None
    assert record["job_id"] and record["input_structure_file"] == str(ensemble.source)
    assert record["temperature"] == 298.15


@pytest.mark.asyncio
async def test_worker_failures_keep_identity_and_artifacts(ensemble):
    ensemble.core.return_value = {
        "status": "failure", "uptake_in_mol_kg": 99,
        "job_id": "wrong", "input_structure_file": "wrong.cif",
        "stdout_path": "/worker/run/raspa.log", "message": "engine failed",
    }
    record, = (await run(ensemble))["results"]
    assert record["uptake_in_mol_kg"] is None
    assert record["job_id"] != "wrong"
    assert record["input_structure_file"] == str(ensemble.source)
    assert record["stdout_path"] == "/worker/run/raspa.log"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["schema", "missing"])
async def test_preparation_failure_after_expansion(ensemble, monkeypatch, failure):
    monkeypatch.setattr(graspa_core, "run_graspa_core", ensemble.real_core)

    def submit(task, count):
        if failure == "schema":
            task.kwargs["job"]["n_cycles"] = 0
        else:
            ensemble.source.unlink()
        return resolved(task.callable(**task.kwargs))

    ensemble.backend.on_submit = submit
    record, = (await run(ensemble))["results"]
    assert record["status"] == "failure"
    assert record["input_structure_file"] == str(ensemble.source)
    assert record["pressure"] == 101325
    assert record["job_id"]


@pytest.mark.asyncio
async def test_real_worker_uses_worker_root_timeout_and_unique_artifacts(ensemble, tmp_path, monkeypatch):
    monkeypatch.setattr(graspa_core, "run_graspa_core", ensemble.real_core)
    monkeypatch.setenv("CHEMGRAPH_GRASPA_EXECUTABLE", sys.executable)
    worker_root = tmp_path / "worker-logs"
    calls = []

    def execute(command, **kwargs):
        calls.append(kwargs)
        kwargs["stdout"].write("Input UnitCells [0] = 1 1 1\nOverall: Average: 12.011,\n")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(graspa_core.subprocess, "run", execute)

    def submit(task, count):
        assert task.kwargs["job"]["output_directory"] == "screen"
        with monkeypatch.context() as worker:
            worker.setenv("CHEMGRAPH_LOG_DIR", str(worker_root))
            return resolved(task.callable(**task.kwargs))

    ensemble.backend.on_submit = submit
    original = ensemble.source.read_bytes()
    records = (await run(
        ensemble, input_structures=[str(ensemble.source)] * 2,
        output_directory="screen", timeout_seconds=13,
    ))["results"]
    assert len({r["run_id"] for r in records}) == 2
    for record in records:
        assert record["status"] == "success"
        assert Path(record["run_dir"]).parent == worker_root / "screen"
        assert Path(record["results_path"]).is_file()
        assert record["cif_path"] != str(ensemble.source)
    assert [call["timeout"] for call in calls] == [13, 13]
    assert ensemble.source.read_bytes() == original


@pytest.mark.asyncio
async def test_async_partial_submission_polling_cancellation_and_reload(ensemble):
    ensemble.backend.is_async_remote = True
    pending = [Future(), Future()]

    def submit(task, count):
        if count == 2:
            raise RuntimeError("submission rejected")
        if count == 3:
            return resolved(error=RuntimeError("worker crashed"))
        return pending[0 if count == 1 else 1]

    ensemble.backend.on_submit = submit
    result = await run(ensemble, input_structures=[str(ensemble.source)] * 4)
    assert result["status"] == "submitted" and result["n_tasks"] == 4
    batch_id = result["batch_id"]
    tracker = ensemble.server._tracker
    batch = tracker._batches[batch_id]
    assert len({task.task_id for task in batch.tasks}) == 4
    # Restart before any poll: locally known submission/crash errors survive.
    reloaded = JobTracker(persist_file=tracker._persist_file)
    failures = reloaded.get_results(batch_id, include_partial=True)["results"]
    assert {record["message"] for record in failures} == {"submission rejected", "worker crashed"}
    pending[0].set_result({"status": "success", "uptake_in_mol_kg": 1.25})
    async with Client(ensemble.server) as client:
        status = job_response(await client.call_tool("check_job_status", {"batch_id": batch_id}))
        assert status["pending_tasks"] == 1 and status["failed_tasks"] == 2
        partial = job_response(await client.call_tool("get_job_results", {"batch_id": batch_id, "include_partial": True}))
        assert len(partial["results"]) == 3
        assert all(r["job_id"] and r["input_structure_file"] == str(ensemble.source) for r in partial["results"])
        cancelled = job_response(await client.call_tool("cancel_job", {"batch_id": batch_id}))
        assert cancelled["cancelled"] == 1
        final = job_response(await client.call_tool("get_job_results", {"batch_id": batch_id}))
        assert final["status"] == "partial" and len(final["results"]) == 4
        assert (await client.call_tool("list_jobs", {})).content
    assert JobTracker(persist_file=tracker._persist_file).get_results(batch_id) == final


@pytest.mark.asyncio
async def test_legacy_future_and_isolated_worker_jsonl(ensemble, tmp_path, monkeypatch):
    with pytest.warns(DeprecationWarning):
        legacy = importlib.reload(importlib.import_module("chemgraph.mcp.graspa_mcp_parsl"))
    monkeypatch.setattr(legacy, "_backend", ensemble.backend)
    params = graspa_input_schema(input_structure_file=str(ensemble.source), adsorbate="H2O")
    assert isinstance(legacy.run_graspa_parsl_app(params), Future)
    invalid = legacy.run_graspa_parsl_app(None)
    assert isinstance(invalid, Future)
    with pytest.raises(TypeError):
        invalid.result()
    ensemble.core.side_effect = [
        {"status": "success", "uptake_in_mol_kg": 1.25}, ValueError("bad CIF"),
    ] * 2
    worker_root = tmp_path / "worker-logs"

    def submit(task, count):
        with monkeypatch.context() as worker:
            worker.setenv("CHEMGRAPH_LOG_DIR", str(worker_root))
            return resolved(task.callable(**task.kwargs))

    ensemble.backend.on_submit = submit
    paths = []
    async with Client(legacy.mcp) as client:
        for _ in range(2):
            response = await client.call_tool("run_graspa_ensemble", {"params": {
                **ensemble.params, "input_structures": [str(ensemble.source)] * 2,
                "output_directory": "summaries",
            }})
            message = response.content[0].text
            assert "Ran 2 tasks (1 successful)" in message
            path = Path(re.search(r"saved to '(.+)'", message).group(1))
            paths.append(path)
            assert path.parent.parent == worker_root / "summaries"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r["status"] for r in records] == ["success", "failure"]
            assert all(r["job_id"] for r in records)
    assert paths[0] != paths[1]
    assert not (ensemble.source.parent / "simulation_results.jsonl").exists()


def test_imports_do_not_initialize_backends_or_optional_dependencies():
    code = """
import sys
from unittest.mock import patch
sys.modules['parsl'] = None
with patch('chemgraph.execution.config.get_backend', side_effect=AssertionError('backend initialized')), patch('chemgraph.execution.config.get_transfer_manager', side_effect=AssertionError('transfer initialized')):
    from chemgraph.mcp import graspa_mcp_hpc, graspa_mcp_parsl
    assert graspa_mcp_hpc.mcp._backend is None
    assert graspa_mcp_parsl._backend is None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("configured", [False, True])
def test_startup_registers_optional_transfer_and_cleans_up(ensemble, monkeypatch, configured):
    from chemgraph.execution import config
    from chemgraph.mcp import server_utils, transfer_tools

    manager = object() if configured else None
    monkeypatch.setattr(config, "get_transfer_manager", lambda: manager)
    register = Mock()
    shutdown = Mock()
    monkeypatch.setattr(transfer_tools, "register_transfer_tools", register)
    monkeypatch.setattr(ensemble.server, "shutdown_backend", shutdown)
    monkeypatch.setattr(server_utils, "run_mcp_server", Mock(side_effect=RuntimeError("server stopped")))
    with pytest.raises(RuntimeError, match="server stopped"):
        hpc.main()
    shutdown.assert_called_once()
    if configured:
        register.assert_called_once_with(ensemble.server, manager)
    else:
        register.assert_not_called()


def test_legacy_backend_is_lazy_and_reused(monkeypatch):
    from chemgraph.execution import config
    from chemgraph.mcp import graspa_mcp_parsl as legacy

    backend = Backend()
    get_backend = Mock(return_value=backend)
    monkeypatch.setattr(legacy, "_backend", None)
    monkeypatch.setattr(config, "get_backend", get_backend)
    monkeypatch.setenv("COMPUTE_SYSTEM", "aurora")
    assert legacy._get_backend() is backend
    assert legacy._get_backend() is backend
    get_backend.assert_called_once_with(backend_name="parsl", system="aurora")


def test_demo_graspa_jobs_preserve_worker_roots_and_input_modes(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "graspa_demo_helpers", Path(__file__).parents[1] / "scripts/demo/_demo_chemistry.py",
    )
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    output = tmp_path / "not-created-on-server"
    jobs = [demo.build_graspa_job("/worker/MOF.CIF", output_dir=output) for _ in range(2)]
    assert jobs[0]["_job_id"] != jobs[1]["_job_id"]
    assert jobs[0]["output_directory"] == str(output)
    assert jobs[0]["output_result_file"] == "raspa.log"
    assert not output.exists()
    prompt = demo.agent_prompt_graspa(remote_directory="~/staged", output_directory="runs")
    assert 'remote_structure_directory="~/staged"' in prompt
    assert 'output_directory="runs"' in prompt
    assert "input_structures=" not in prompt
    assert "input_structures=" in demo.agent_prompt_graspa(["/shared/MOF.CIF"])
    with pytest.raises(ValueError, match="exclusively"):
        demo.agent_prompt_graspa(["/shared/MOF.CIF"], remote_directory="~/staged")
    demo.abort_if_graspa_unsupported("graspa", "local")
