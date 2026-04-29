"""Backend-agnostic XANES/FDMNES MCP server.

Replaces ``xanes_mcp_parsl.py`` by using the :mod:`chemgraph.execution`
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
    resolve_structure_files,
    write_results_jsonl,
)
from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.schemas.xanes_schema import (
    mp_query_schema,
    xanes_input_schema,
    xanes_input_schema_ensemble,
)

logger = logging.getLogger(__name__)

# ── Initialise execution backend ────────────────────────────────────────
backend = get_backend()

# ── MCP server ──────────────────────────────────────────────────────────
mcp = FastMCP(
    name="ChemGraph XANES Tools",
    instructions="""
        You expose tools for running XANES/FDMNES simulations.
        The available tools are:
        1. run_xanes_single: run a single FDMNES calculation for one structure.
        2. run_xanes_ensemble: run FDMNES calculations over multiple structures
           using the configured execution backend.
        3. fetch_mp_structures: fetch optimized structures from Materials Project.
        4. plot_xanes: generate normalized XANES plots for completed calculations.

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they occur.
        - Keep responses compact -- full results are in the output directories.
        - When returning paths, use absolute paths.
        - Energies are in eV.
    """,
)


@mcp.tool(
    name="run_xanes_single",
    description="Run a single XANES/FDMNES calculation for one input structure.",
)
def run_xanes_single(params: xanes_input_schema):
    """Run a single FDMNES calculation using the core engine."""
    from chemgraph.tools.xanes_tools import run_xanes_core

    return run_xanes_core(params)


def _xanes_post_fn(meta: dict, _result) -> dict:
    """Post-process a completed FDMNES task: extract convergence data."""
    from chemgraph.tools.xanes_tools import extract_conv

    try:
        conv_data = extract_conv(meta["run_dir"])
        return {
            **meta,
            "status": "success",
            "n_conv_files": len(conv_data),
        }
    except Exception as e:
        return {
            **meta,
            "status": "failure",
            "error_type": type(e).__name__,
            "message": f"Post-processing failed: {e}",
        }


@mcp.tool(
    name="run_xanes_ensemble",
    description="Run an ensemble of XANES/FDMNES calculations using the configured backend.",
)
async def run_xanes_ensemble(params: xanes_input_schema_ensemble):
    """Run ensemble XANES calculations over all structure files.

    For each structure file:
    1. Reads the structure via ASE.
    2. Creates FDMNES input files in a per-structure subdirectory.
    3. Submits a shell task to run FDMNES.
    4. Gathers results and writes a JSONL summary log.

    Parameters
    ----------
    params : xanes_input_schema_ensemble
        Input parameters for the ensemble calculation.
    """
    from ase.io import read as ase_read

    from chemgraph.tools.xanes_tools import write_fdmnes_input

    structure_files, output_dir = resolve_structure_files(
        params.input_structures,
        extensions={".cif", ".xyz", ".poscar"},
    )

    # Create a batch runs directory
    runs_dir = output_dir / "fdmnes_batch_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    fdmnes_exe = params.fdmnes_exe

    pending_tasks = []

    for i, struct_path in enumerate(structure_files):
        run_dir = runs_dir / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Read structure and write FDMNES inputs
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

        # Submit shell task
        task = TaskSpec(
            task_id=f"xanes_{struct_path.stem}_{i}",
            task_type="shell",
            command=f'cd "{run_dir}" && "{fdmnes_exe}"',
            working_dir=str(run_dir),
            stdout=str(run_dir / "fdmnes_stdout.txt"),
            stderr=str(run_dir / "fdmnes_stderr.txt"),
        )
        fut = backend.submit(task)

        task_meta = {
            "structure": struct_path.name,
            "run_dir": str(run_dir),
            "z_absorber": z_abs,
        }
        pending_tasks.append((task_meta, fut))

    results = await gather_futures(pending_tasks, post_fn=_xanes_post_fn)

    summary_log_path = output_dir / "xanes_results.jsonl"
    success_count, total_count = write_results_jsonl(results, summary_log_path)

    return (
        f"Ensemble execution completed. Ran {total_count} tasks "
        f"({success_count} successful). "
        f"Detailed results appended to '{summary_log_path}'."
    )


@mcp.tool(
    name="fetch_mp_structures",
    description="Fetch optimized structures from Materials Project.",
)
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


@mcp.tool(
    name="plot_xanes",
    description="Generate normalized XANES plots for completed FDMNES calculations.",
)
def plot_xanes(runs_dir: str):
    """Generate XANES plots for all completed runs in a directory.

    Parameters
    ----------
    runs_dir : str
        Path to the ``fdmnes_batch_runs`` directory containing ``run_*``
        subdirectories with FDMNES outputs.
    """
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


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9007)
