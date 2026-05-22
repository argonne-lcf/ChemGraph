"""Backend-agnostic MACE MCP server.

Uses :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP` so that tool
functions are plain computation — the framework handles backend
submission, future resolution, and async job tracking.

Nothing is initialised at import time so that worker subprocesses
(e.g. EnsembleLauncher) can safely re-import this module.
"""

from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.schemas.mace_parsl_schema import mace_input_schema
from chemgraph.tools.parsl_tools import extract_output_json, run_mace_core

mcp = CGFastMCP(
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


@mcp.tool(
    name="run_mace_single",
    description="Run a single MACE calculation",
)
def run_mace_single(params: mace_input_schema):
    """Run a single MACE calculation on the execution backend."""
    import sys

    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        return run_mace_core(params)
    finally:
        sys.stdout = old_stdout


@mcp.ensemble_tool(
    name="run_mace_ensemble",
    description="Run an ensemble of MACE calculations for multiple inputs.",
)
def _run_mace_worker(params: mace_input_schema):
    return run_mace_core(params)


mcp.add_tool(
    extract_output_json,
    name="extract_output_json",
    description="Load output from a JSON file.",
)


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    mcp.init_backend()

    try:
        run_mcp_server(mcp, default_port=9004)
    finally:
        mcp.shutdown_backend()
