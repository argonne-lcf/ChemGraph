"""Backend-agnostic gRASPA MCP server.

Replaces ``graspa_mcp_parsl.py`` by using the :mod:`chemgraph.execution`
abstraction layer.  The execution backend (Parsl, EnsembleLauncher,
local) is selected at startup via ``config.toml`` or the
``CHEMGRAPH_EXECUTION_BACKEND`` environment variable.
"""

import logging
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from chemgraph.execution import TaskSpec, get_backend
from chemgraph.execution.utils import (
    gather_futures,
    make_per_structure_output,
    resolve_structure_files,
    write_results_jsonl,
)
from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.schemas.graspa_schema import graspa_input_schema_ensemble

logger = logging.getLogger(__name__)

# ── Initialise execution backend ────────────────────────────────────────
backend = get_backend()

# ── MCP server ──────────────────────────────────────────────────────────
mcp = FastMCP(
    name="ChemGraph Graspa Tools",
    instructions="""
        You expose tools for running graspa simulations and reading their results.
        The available tools are:
        1. run_graspa_ensemble: run graspa calculations over all structures in a
           directory using the configured execution backend.

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they occur.
        - Keep responses compact -- full results are written to the output files
          defined in the schemas.
        - When returning paths, use absolute paths.
        - Energies are in eV and wall times are in seconds.
    """,
)


def _run_graspa_single(job: dict) -> dict:
    """Execute a single gRASPA simulation (runs on the worker)."""
    from chemgraph.schemas.graspa_schema import graspa_input_schema
    from chemgraph.tools.graspa_tools import run_graspa_core

    params = graspa_input_schema(**job) if isinstance(job, dict) else job
    return run_graspa_core(params)


@mcp.tool(
    name="run_graspa_ensemble",
    description="Run an ensemble of graspa calculations for multiple input files.",
)
async def run_graspa_ensemble(
    params: graspa_input_schema_ensemble,
):
    """Run an ensemble of gRASPA calculations over all structure files
    using the configured execution backend.

    Parameters
    ----------
    params : graspa_input_schema_ensemble
        Input parameters for the ensemble of gRASPA calculations.
    """
    structure_files, output_dir = resolve_structure_files(
        params.input_structures,
        extensions={".cif"},
    )

    # Base output file name
    base_output = Path(params.output_result_file).resolve()

    pending_tasks = []

    for struct_path in structure_files:
        mof_name = struct_path.stem
        for condition in params.conditions:
            per_struct_output = make_per_structure_output(struct_path, base_output)
            job = {
                "input_structure_file": str(struct_path),
                "output_result_file": str(per_struct_output),
                "temperature": condition.temperature,
                "pressure": condition.pressure,
                "adsorbate": params.adsorbate,
                "n_cycles": params.n_cycles,
            }

            task = TaskSpec(
                task_id=f"graspa_{mof_name}_{condition.temperature}K_{condition.pressure}Pa",
                task_type="python",
                callable=_run_graspa_single,
                kwargs={"job": job},
            )
            fut = backend.submit(task)

            task_meta = {
                "structure": mof_name,
                "temperature": condition.temperature,
                "pressure": condition.pressure,
            }
            pending_tasks.append((task_meta, fut))

    results = await gather_futures(pending_tasks)

    summary_log_path = output_dir / "simulation_results.jsonl"
    success_count, total_count = write_results_jsonl(results, summary_log_path)

    return (
        f"Ensemble execution completed. Ran {total_count} tasks "
        f"({success_count} successful). "
        f"Detailed results appended to '{summary_log_path}'."
    )


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9001)
