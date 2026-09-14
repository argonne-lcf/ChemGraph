"""Hermetic coverage of login-node submission and compute-node ASE execution."""

import asyncio
from concurrent.futures import Future
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import pytest
from ase.calculators.emt import EMT

from chemgraph.execution.parsl_backend import ParslBackend
from chemgraph.tools import ase_core, ase_runner


@pytest.fixture
def calculation(tmp_path, monkeypatch):
    xyz = tmp_path / "hydrogen.xyz"
    xyz.write_text("2\nHydrogen smoke test\nH 0 0 0\nH 0 0 1.0\n")
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(ase_core, "_ensure_ase_core_file_log", lambda: None)
    return {
        "input_structure_file": str(xyz),
        "output_results_file": str(tmp_path / "result.json"),
        "driver": "opt",
        "calculator": {"calculator_type": "emt"},
        "fmax": 0.01,
        "steps": 100,
    }


def pbs_environment(tmp_path, monkeypatch):
    nodes = tmp_path / "nodes"
    nodes.write_text(socket.gethostname() + "\n")
    monkeypatch.setenv("PBS_NODEFILE", str(nodes))
    monkeypatch.setenv("PBS_JOBID", "123.polaris-pbs.example")


class DipoleEMT(EMT):
    """Synthetic dipoles for exercising IR export, not a scientific model."""

    def get_dipole_moment(self, atoms=None):
        return 0.1 * (atoms.positions[1] - atoms.positions[0])


@pytest.mark.parametrize("driver", ["opt", "vib", "ir", "thermo"])
def test_all_drivers_preserve_results_and_artifacts(
    calculation, tmp_path, monkeypatch, driver
):
    calculation["driver"] = driver
    if driver == "ir":
        monkeypatch.setattr(
            ase_core, "load_calculator", lambda _: (DipoleEMT(), {}, None)
        )
    request = tmp_path / "input.json"
    request.write_text(json.dumps(calculation))
    assert ase_runner.run_calculation(request) == 0
    result = json.loads((tmp_path / "result.json").read_text())
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert result["success"] and result["converged"]
    assert summary["potential_energy"] == result["potential_energy"]
    assert summary["energy_unit"] == "eV"
    assert summary["optimization_steps"] > 0
    assert (tmp_path / "final.xyz").is_file()
    assert (tmp_path / "hydrogen_opt.traj").is_file()
    if driver in {"vib", "ir", "thermo"}:
        assert result["vibrational_frequencies"]["frequencies"]
        assert (tmp_path / "frequencies_hydrogen.csv").is_file()
    if driver == "ir":
        assert result["ir_data"]
        assert (tmp_path / "ir_spectrum_hydrogen.png").is_file()
    if driver == "thermo":
        assert result["thermochemistry"]
    before = (tmp_path / "run_summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        ase_runner.run_calculation(request)
    assert (tmp_path / "run_summary.json").read_bytes() == before


@pytest.mark.parametrize(
    "failure", ["steps", "missing_structure", "missing_model", "cuda", "dipoles"]
)
def test_failures_do_not_report_scientific_success(
    calculation, tmp_path, monkeypatch, failure
):
    if failure == "steps":
        calculation["steps"] = 0
    elif failure == "missing_structure":
        calculation["input_structure_file"] = str(tmp_path / "missing.xyz")
    elif failure in {"missing_model", "cuda"}:
        model = tmp_path / "model"
        if failure == "cuda":
            model.write_text("Model must not be loaded without CUDA")
        calculation["calculator"] = {
            "calculator_type": "mace_polar",
            "model": str(model),
            "device": "cuda",
        }
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(
            ase_core, "run_ase_core", lambda _: pytest.fail("Calculator must not load")
        )
    else:
        calculation["driver"] = "ir"
    request = tmp_path / "input.json"
    request.write_text(json.dumps(calculation))
    assert ase_runner.run_calculation(request) == (2 if failure == "steps" else 1)
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["status"] != "success"
    if failure == "steps":
        assert summary["converged"] is False
        assert (tmp_path / "final.xyz").exists()


@pytest.mark.parametrize("context", ["missing", "wrong_host"])
def test_guard_rejects_login_node_before_calculation(tmp_path, monkeypatch, context):
    monkeypatch.delenv("PBS_JOBID", raising=False)
    if context == "wrong_host":
        pbs_environment(tmp_path, monkeypatch)
        (tmp_path / "nodes").write_text("another-compute-node\n")
    with pytest.raises(RuntimeError):
        ase_runner.run_calculation("nonexistent-input.json", require_pbs=True)


def test_worker_subprocess_isolates_artifacts_and_rejects_duplicates(
    calculation, tmp_path, monkeypatch
):
    pbs_environment(tmp_path, monkeypatch)
    parent_log_dir = os.environ["CHEMGRAPH_LOG_DIR"]
    results = []
    for name in ("first", "second"):
        job = {
            **calculation,
            "driver": "vib",
            "output_results_file": str(tmp_path / name / "result.json"),
        }
        result = ase_runner.run_pbs_calculation(job)
        assert result["status"] == "success", result
        assert result["pbs_job_id"] == "123.polaris-pbs.example"
        assert (tmp_path / name / "frequencies_hydrogen.csv").exists()
        results.append(result)
    before = Path(results[-1]["summary_file"]).read_bytes()
    assert ase_runner.run_pbs_calculation(job)["error_type"] == "FileExistsError"
    assert Path(results[-1]["summary_file"]).read_bytes() == before
    assert os.environ["CHEMGRAPH_LOG_DIR"] == parent_log_dir
    assert not (tmp_path / "frequencies_hydrogen.csv").exists()


def test_parsl_pbs_configuration_and_option_forwarding(tmp_path, monkeypatch):
    parsl = pytest.importorskip("parsl")
    from parsl.providers import LocalProvider, PBSProProvider
    from chemgraph.hpc_configs.loader import load_parsl_config

    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("PBS_NODEFILE", raising=False)
    captures = []
    monkeypatch.setattr(parsl, "load", captures.append)
    backend = ParslBackend()
    backend.initialize(
        system="polaris",
        allocation_mode="pbs",
        account="test",
        worker_init="source '/shared/env setup.sh'",
        run_dir=str(tmp_path / "space here"),
        address="127.0.0.1",
        max_workers_per_node=4,
    )
    config = captures[0]
    executor = config.executors[0]
    provider = executor.provider
    assert backend.is_async_remote
    assert isinstance(provider, PBSProProvider)
    assert (provider.init_blocks, provider.min_blocks, provider.max_blocks) == (0, 0, 1)
    assert provider.account == "test" and provider.queue == "debug"
    assert executor.max_workers_per_node == 4
    assert str(tmp_path / "space here") in provider.worker_init
    assert "--depth=64" in provider.launcher.overrides
    assert isinstance(load_parsl_config("polaris").executors[0].provider, LocalProvider)
    assert not ParslBackend().is_async_remote
    with pytest.raises(TypeError):
        load_parsl_config("polaris", allocation_mode="pbs", address="127.0.0.1")
    with pytest.raises(ValueError, match="Polaris only"):
        load_parsl_config("local", allocation_mode="pbs")
    with pytest.raises(ValueError, match="allocation_mode"):
        load_parsl_config("polaris", allocation_mode="typo")


@pytest.mark.asyncio
async def test_mcp_returns_pending_batch_and_tracks_same_calculation(
    tmp_path, monkeypatch, calculation
):
    from fastmcp import Client
    from chemgraph.execution.job_tracker import JobTracker
    from chemgraph.mcp import ase_mcp_hpc as ase_server
    from chemgraph.mcp.cg_fastmcp import CGFastMCP

    server = CGFastMCP(name="PBS ASE test")
    server.init_backend()
    backend = ParslBackend()
    backend._initialized = True
    backend._is_async_remote = True
    future = Future()
    submitted = []
    backend._python_app = lambda fn, args, kwargs: (
        submitted.append((fn, kwargs)),
        future,
    )[1]
    backend._executors = [
        SimpleNamespace(
            label="htex",
            blocks_to_job_id={"0": "123.pbs"},
            status_facade={
                "0": SimpleNamespace(
                    state=SimpleNamespace(name="PENDING"), message="Queued"
                )
            },
        )
    ]
    server._backend = backend
    server._tracker = JobTracker(tmp_path / "jobs.json")
    monkeypatch.setattr(ase_server, "mcp", server)
    monkeypatch.setattr(ase_server, "_PBS_WORKERS", True)
    server.set_pre_submit_hook(ase_server._ase_transport_hook)
    server.tool(name="run_ase_single")(ase_server.run_ase_single)
    server.add_tool(ase_server.get_execution_status, name="get_execution_status")

    def payload(response):
        return (
            response.data
            if response.data is not None
            else json.loads(response.content[0].text)
        )

    async with Client(server) as client:
        response = await asyncio.wait_for(
            client.call_tool("run_ase_single", {"params": calculation}), timeout=5
        )
        batch = payload(response)["batch_id"]
        assert payload(response)["status"] == "submitted"
        assert submitted[0][1]["job"]["_pbs_worker"] is True
        for _ in range(2):
            status = await client.call_tool("check_job_status", {"batch_id": batch})
            assert payload(status)["status"] != "completed"
        allocation = await client.call_tool("get_execution_status", {})
        assert payload(allocation)["allocations"][0]["scheduler_job_id"] == "123.pbs"
        future.set_result(
            {"status": "success", "converged": True, "potential_energy": -1.0}
        )
        result = await client.call_tool("get_job_results", {"batch_id": batch})
        assert payload(result)["results"][0]["converged"] is True
    assert len(submitted) == 1


def test_pbs_ensemble_uses_distinct_artifact_directories(tmp_path, monkeypatch):
    from chemgraph.mcp import ase_mcp_hpc as server
    from chemgraph.schemas.ase_input import ase_input_schema_ensemble

    structures = tmp_path / "structures"
    structures.mkdir()
    for name in ("water.xyz", "water.cif"):
        (structures / name).touch()
    monkeypatch.setattr(server, "_PBS_WORKERS", True)
    jobs = server._expand_ase_ensemble(
        ase_input_schema_ensemble(
            input_structure_directory=str(structures),
            output_results_file=str(tmp_path / "results" / "output.json"),
            driver="ir",
            calculator={"calculator_type": "emt"},
        )
    )
    assert len({Path(job["output_results_file"]).parent for job in jobs}) == 2
    assert {job["driver"] for job in jobs} == {"ir"}
