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

import json
import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from chemgraph.execution import TaskSpec, get_backend
from chemgraph.execution.utils import (
    gather_futures,
    make_per_structure_output,
    resolve_structure_files,
)
from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.tools.parsl_tools import (
    mace_input_schema,
    mace_input_schema_ensemble,
    run_mace_core,
)

logger = logging.getLogger(__name__)

# ── Initialise execution backend ────────────────────────────────────────
backend = get_backend()

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

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they occur.
        - Keep responses compact -- full results are written to the output files
          defined in the schemas.
        - When returning paths, use absolute paths.
        - Energies are in eV and wall times are in seconds.
    """,
)


def _run_mace_single(job: dict) -> dict:
    """Execute a single MACE simulation (runs on the worker)."""
    from chemgraph.tools.parsl_tools import mace_input_schema, run_mace_core

    params = mace_input_schema(**job) if isinstance(job, dict) else job
    return run_mace_core(params)


@mcp.tool(
    name="run_mace_single",
    description="Run a single MACE calculation",
)
def run_mace_single(params: mace_input_schema):
    return run_mace_core(params)


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

    results = await gather_futures(pending_tasks, post_fn=_mace_post_fn)

    return {
        "status": "success",
        "n_structures": len(structure_files),
        "results": results,
    }


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
