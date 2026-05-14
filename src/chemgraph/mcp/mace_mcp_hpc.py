"""Backend-agnostic MACE MCP server.

Replaces ``mace_mcp_parsl.py`` by using the :mod:`chemgraph.execution`
abstraction layer.  The execution backend (Parsl, EnsembleLauncher,
local) is selected at startup via ``config.toml`` or the
``CHEMGRAPH_EXECUTION_BACKEND`` environment variable.

Key improvements over the original:
- No hardcoded Polaris config or user-specific conda paths.
- Ensemble tool is now async (non-blocking event loop).
- Uses shared utilities for structure resolution and result gathering.
"""

import asyncio
import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from chemgraph.execution import TaskSpec, get_backend
from chemgraph.execution.job_tracker import JobTracker
from chemgraph.execution.utils import (
    make_per_structure_output,
    resolve_structure_files,
    submit_or_gather,
)
from chemgraph.mcp.job_tools import register_job_tools
from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.tools.parsl_tools import (
    mace_input_schema,
    mace_input_schema_ensemble,
)

logger = logging.getLogger(__name__)

# ── Initialise execution backend ────────────────────────────────────────
backend = get_backend()
tracker = JobTracker()

# ── MCP server ──────────────────────────────────────────────────────────
mcp = FastMCP(
    name="ChemGraph MACE Tools",
    instructions="""
        You expose tools for running MACE simulations and reading their results.
        The available tools are:
        1. run_mace_single: run a single MACE calculation using the specified
           input schema.
        2. run_mace_ensemble: run MACE calculations over all structures in a
           directory using the configured execution backend.
        3. extract_output_json: load simulation results from a JSON file.
        4. check_job_status: check progress of a submitted HPC job batch.
        5. get_job_results: retrieve results from a completed job batch.
        6. list_jobs: list all tracked job batches.
        7. cancel_job: cancel pending tasks in a job batch.

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they occur.
        - Keep responses compact -- full results are written to the output files
          defined in the schemas.
        - When returning paths, use absolute paths.
        - Energies are in eV and wall times are in seconds.
        - When a tool returns status='submitted' with a batch_id, use
          check_job_status to poll for progress before calling get_job_results.
    """,
)
register_job_tools(mcp, tracker, backend)


def _run_mace_single(job: dict) -> dict:
    """Execute a single MACE simulation (runs on the worker).

    When the ``job`` dict contains an ``inline_structure`` key (with
    ``numbers``, ``positions``, and optional ``cell``/``pbc``), the
    structure is materialised as a temporary XYZ file on the worker
    filesystem before running MACE.  This allows local-agent /
    remote-worker workflows where the original file only exists on the
    submitting machine.
    """
    import os
    import tempfile

    from chemgraph.tools.parsl_tools import mace_input_schema, run_mace_core

    inline = job.pop("inline_structure", None)
    if inline is not None:
        from ase import Atoms
        from ase.io import write as ase_write

        atoms = Atoms(
            numbers=inline["numbers"],
            positions=inline["positions"],
            cell=inline.get("cell"),
            pbc=inline.get("pbc"),
        )
        tmpdir = tempfile.mkdtemp(prefix="chemgraph_mace_")
        xyz_path = os.path.join(tmpdir, "structure.xyz")
        ase_write(xyz_path, atoms)
        job["input_structure_file"] = xyz_path

        if not os.path.isabs(job.get("output_result_file", "")):
            job["output_result_file"] = os.path.join(
                tmpdir, job.get("output_result_file", "output.json")
            )

    params = mace_input_schema(**job) if isinstance(job, dict) else job
    result = run_mace_core(params)

    # Embed full output JSON when running with inline structure so the
    # caller does not need to read a file on the remote filesystem.
    if inline is not None:
        out_file = job.get("output_result_file", "")
        if os.path.isfile(out_file):
            import json as _json

            with open(out_file, "r") as fh:
                result["full_output"] = _json.load(fh)

    return result


@mcp.tool(
    name="run_mace_single",
    description="Run a single MACE calculation",
)
async def run_mace_single(params: mace_input_schema):
    """Run a single MACE calculation using the configured execution backend."""
    job = params.model_dump()

    # Read the local structure file and embed it so the job is
    # self-contained and can run on any worker (local or remote).
    input_file = job.get("input_structure_file")
    if input_file and os.path.isfile(input_file):
        from ase.io import read as ase_read

        from chemgraph.tools.ase_core import atoms_to_atomsdata

        atoms = ase_read(input_file)
        atomsdata = atoms_to_atomsdata(atoms)
        job["inline_structure"] = atomsdata.model_dump()

    task = TaskSpec(
        task_id="mace_single",
        task_type="python",
        callable=_run_mace_single,
        kwargs={"job": job},
    )
    fut = backend.submit(task)

    if backend.is_async_remote:
        task_meta = {"task_id": "mace_single"}
        return await submit_or_gather(
            backend, [(task_meta, fut)], tracker, "run_mace_single"
        )

    return await asyncio.wrap_future(fut)


def _mace_post_fn(meta: dict, result) -> dict:
    """Post-process a completed MACE task."""
    status = result.get("status", "unknown") if isinstance(result, dict) else "success"
    energy = result.get("single_point_energy") if isinstance(result, dict) else None
    return {
        "structure": meta["structure"],
        "output_result_file": meta["output_result_file"],
        "status": status,
        "single_point_energy": energy,
        "raw_result": result,
    }


@mcp.tool(
    name="run_mace_ensemble",
    description="Run an ensemble of MACE calculations",
)
async def run_mace_ensemble(params: mace_input_schema_ensemble):
    """Run an ensemble of MACE calculations over all structure files in a
    directory using the configured execution backend.

    Parameters
    ----------
    params : mace_input_schema_ensemble
        Input parameters for the ensemble of MACE calculations.

    Returns
    -------
    dict
        Summary of all jobs with minimal per-job results.
    """
    structure_files, _output_dir = resolve_structure_files(
        params.input_structure_directory,
    )

    # Base output file name used as a pattern for per-structure outputs
    base_output = Path(params.output_result_file)

    pending_tasks = []
    for struct_path in structure_files:
        per_struct_output = make_per_structure_output(struct_path, base_output)

        job = {
            "input_structure_file": str(struct_path),
            "output_result_file": str(per_struct_output),
            "driver": params.driver,
            "model": params.model,
            "device": params.device,
            "temperature": params.temperature,
            "pressure": params.pressure,
            "fmax": params.fmax,
            "steps": params.steps,
            "optimizer": params.optimizer,
        }

        # Embed structure data so the job works on remote workers that
        # cannot access the local filesystem.
        if struct_path.is_file():
            from ase.io import read as ase_read

            from chemgraph.tools.ase_core import atoms_to_atomsdata

            atoms = ase_read(str(struct_path))
            atomsdata = atoms_to_atomsdata(atoms)
            job["inline_structure"] = atomsdata.model_dump()

        task = TaskSpec(
            task_id=f"mace_{struct_path.stem}",
            task_type="python",
            callable=_run_mace_single,
            kwargs={"job": job},
        )
        fut = backend.submit(task)

        task_meta = {
            "structure": struct_path.name,
            "output_result_file": str(per_struct_output),
        }
        pending_tasks.append((task_meta, fut))

    result = await submit_or_gather(
        backend, pending_tasks, tracker, "run_mace_ensemble",
        post_fn=_mace_post_fn,
    )

    if result["status"] == "completed":
        return {
            "status": "success",
            "n_structures": len(structure_files),
            "results": result["results"],
        }

    # Async remote: return submission confirmation
    result["n_structures"] = len(structure_files)
    return result


@mcp.tool(
    name="extract_output_json",
    description="Load output from a JSON file.",
)
def extract_output_json(json_file: str) -> dict:
    """Load simulation results from a JSON file produced by run_ase.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing ASE simulation results.

    Returns
    -------
    dict
        Parsed results from the JSON file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9004)
