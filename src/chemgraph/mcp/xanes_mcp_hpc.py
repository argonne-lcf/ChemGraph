"""Backend-agnostic XANES/FDMNES MCP server.

Uses :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`. Tool functions are
plain computation -- the framework handles backend submission, future
resolution, and async job tracking.

The ensemble expander runs server-side and prepares per-structure
FDMNES input files in ``runs_dir``; the worker (which runs on the
backend) executes FDMNES via subprocess and extracts convergence data.
This assumes the server and worker share a filesystem (true for any
Globus Compute endpoint on the same HPC where the MCP server runs;
Globus Transfer staging is a separate concern).

Nothing requiring the backend is initialised at import time so worker
subprocesses (EnsembleLauncher, Globus Compute) can re-import this
module safely.
"""

import logging
from pathlib import Path

from chemgraph.execution.config import get_transfer_manager
from chemgraph.execution.utils import resolve_structure_files
from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.mcp.transfer_tools import register_transfer_tools
from chemgraph.mcp.xanes_worker import _xanes_ensemble_worker, run_xanes_single
from chemgraph.schemas.xanes_schema import (
    mp_query_schema,
    xanes_input_schema_ensemble,
)

logger = logging.getLogger(__name__)

_JOBS_FILE = Path("~/.chemgraph/xanes_jobs.json").expanduser()

mcp = CGFastMCP(
    name="ChemGraph XANES Tools",
    instructions="""
        You expose tools for running XANES/FDMNES simulations.
        The available tools are:
        1. run_xanes_single: run a single FDMNES calculation for one structure.
        2. run_xanes_ensemble: run FDMNES calculations over multiple structures
           using the configured execution backend.
        3. fetch_mp_structures: fetch optimized structures from Materials Project.
        4. plot_xanes: generate normalized XANES plots for completed calculations.
        5. check_job_status / get_job_results / list_jobs / cancel_job: HPC
           job batch management. Job state persists across sessions.
        6. transfer_files / check_transfer_status / list_remote_files
           (when Globus Transfer is configured): stage input files on the
           remote HPC filesystem before running ensembles.

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they occur.
        - Keep responses compact -- full results are in the output directories.
        - When returning paths, use absolute paths.
        - Energies are in eV.
        - When a tool returns status='submitted' with a batch_id, call
          get_job_results(batch_id) to retrieve results.  If the job is
          still pending, report the batch_id to the user so they can
          check later.  Job state is persisted across sessions -- the
          user can call list_jobs or get_job_results in a future session
          to retrieve results.
    """,
)


mcp.tool(
    name="run_xanes_single",
    description="Run a single XANES/FDMNES calculation for one input structure.",
)(
    run_xanes_single
)


# ── Ensemble fanout ────────────────────────────────────────────────────


def _expand_xanes_ensemble(params: xanes_input_schema_ensemble) -> list[dict]:
    """Server-side expansion: prepare per-structure run dirs and return
    one item per structure for the worker to execute."""
    from ase.io import read as ase_read

    from chemgraph.tools.xanes_tools import write_fdmnes_input

    structure_files, output_dir = resolve_structure_files(
        params.input_structures,
        extensions={".cif", ".xyz", ".poscar"},
    )

    runs_dir = output_dir / "fdmnes_batch_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    items: list[dict] = []
    for i, struct_path in enumerate(structure_files):
        run_dir = runs_dir / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)

        atoms = ase_read(str(struct_path))
        z_abs = (
            params.z_absorber
            if params.z_absorber is not None
            else int(max(atoms.get_atomic_numbers()))
        )

        write_fdmnes_input(
            ase_atoms=atoms,
            z_absorber=z_abs,
            input_file_dir=run_dir,
            radius=params.radius,
            magnetism=params.magnetism,
        )

        items.append(
            {
                "structure": struct_path.name,
                "run_dir": str(run_dir),
                "z_absorber": z_abs,
                "fdmnes_exe": params.fdmnes_exe,
            }
        )

    return items


@mcp.schema_fanout_tool(
    name="run_xanes_ensemble",
    description=(
        "Run FDMNES/XANES calculations over every structure in an input "
        "directory (or list of files). Each structure is prepared "
        "server-side and submitted to the configured execution backend."
    ),
    worker=_xanes_ensemble_worker,
)
def run_xanes_ensemble(params: xanes_input_schema_ensemble) -> list[dict]:
    return _expand_xanes_ensemble(params)


# ── Orchestration tools (no backend involvement) ───────────────────────


def fetch_mp_structures(params: mp_query_schema):
    """Fetch structures from Materials Project and save as CIF files and pickle database."""
    from chemgraph.tools.xanes_tools import (
        _get_data_dir,
        fetch_materials_project_data,
    )

    data_dir = _get_data_dir()
    result = fetch_materials_project_data(params, data_dir)
    return {
        "status": "success",
        "n_structures": result["n_structures"],
        "chemsys": params.chemsys,
        "output_dir": str(data_dir),
        "structure_files": result["structure_files"],
        "pickle_file": result["pickle_file"],
    }


def plot_xanes(runs_dir: str):
    """Generate XANES plots for all completed runs in a directory."""
    from chemgraph.tools.xanes_tools import (
        _get_data_dir,
        plot_xanes_results,
    )

    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise ValueError(f"'{runs_dir}' is not a valid directory.")

    data_dir = _get_data_dir()
    result = plot_xanes_results(data_dir, runs_path)
    return {
        "status": "success",
        "n_plots": result["n_plots"],
        "n_failed": result["n_failed"],
        "plot_files": result["plot_files"],
        "failed": result["failed"],
    }


mcp.add_tool(
    fetch_mp_structures,
    name="fetch_mp_structures",
    description="Fetch optimized structures from Materials Project.",
)
mcp.add_tool(
    plot_xanes,
    name="plot_xanes",
    description="Generate normalized XANES plots for completed FDMNES calculations.",
)


# ── Globus Transfer (registered only when configured) ──────────────────

_transfer_manager = get_transfer_manager()
if _transfer_manager is not None:
    register_transfer_tools(mcp, _transfer_manager)
    logger.info("Registered Globus Transfer tools on XANES MCP server.")


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})

    try:
        run_mcp_server(mcp, default_port=9007)
    finally:
        mcp.shutdown_backend()
