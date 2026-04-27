import asyncio
import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import parsl
from parsl import bash_app

from chemgraph.mcp.server_utils import load_parsl_config, run_mcp_server
from chemgraph.schemas.xanes_schema import (
    xanes_input_schema,
    xanes_input_schema_ensemble,
    mp_query_schema,
)


@bash_app
def run_fdmnes_parsl_app(
    run_dir: str,
    fdmnes_exe: str,
    stdout=None,
    stderr=None,
):
    """Parsl bash_app that runs FDMNES in a prepared input directory.

    Parameters
    ----------
    run_dir : str
        Path to the directory containing fdmfile.txt and fdmnes_in.txt.
    fdmnes_exe : str
        Path to the FDMNES executable.
    """
    return f'cd "{run_dir}" && "{fdmnes_exe}"'


# Load Parsl config at module level (same pattern as graspa_mcp_parsl.py)
target_system = os.getenv("COMPUTE_SYSTEM", "polaris")
parsl.load(load_parsl_config(target_system))

# Start MCP server
mcp = FastMCP(
    name="ChemGraph XANES Tools",
    instructions="""
        You expose tools for running XANES/FDMNES simulations.
        The available tools are:
        1. run_xanes_single: run a single FDMNES calculation for one structure.
        2. run_xanes_ensemble: run FDMNES calculations over multiple structures
           using Parsl for parallel execution.
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
    from chemgraph.tools.xanes_core import run_xanes_core

    return run_xanes_core(params)


@mcp.tool(
    name="run_xanes_ensemble",
    description="Run an ensemble of XANES/FDMNES calculations using Parsl.",
)
async def run_xanes_ensemble(params: xanes_input_schema_ensemble):
    """Run ensemble XANES calculations over all structure files using Parsl.

    For each structure file:
    1. Reads the structure via ASE.
    2. Creates FDMNES input files in a per-structure subdirectory.
    3. Submits a Parsl bash_app to run FDMNES.
    4. Gathers results and writes a JSONL summary log.

    Parameters
    ----------
    params : xanes_input_schema_ensemble
        Input parameters for the ensemble calculation.
    """
    from ase.io import read as ase_read

    from chemgraph.tools.xanes_core import (
        write_fdmnes_input,
        extract_conv,
    )

    input_source = params.input_structures
    structure_files: list[Path] = []
    output_dir: Path = Path.cwd()

    if isinstance(input_source, list):
        structure_files = [Path(p) for p in input_source]
        missing = [p for p in structure_files if not p.exists()]
        if missing:
            raise ValueError(f"The following input files are missing: {missing}")
        if structure_files:
            output_dir = structure_files[0].parent
    else:
        input_dir = Path(input_source)
        if not input_dir.is_dir():
            raise ValueError(f"'{input_dir}' is not a valid directory.")
        structure_files = sorted(
            p for p in input_dir.iterdir() if p.suffix in {".cif", ".xyz", ".poscar"}
        )
        output_dir = input_dir

    if not structure_files:
        raise ValueError("No structure files found to simulate.")

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

        # Submit Parsl task
        fut = run_fdmnes_parsl_app(
            run_dir=str(run_dir),
            fdmnes_exe=fdmnes_exe,
            stdout=str(run_dir / "fdmnes_stdout.txt"),
            stderr=str(run_dir / "fdmnes_stderr.txt"),
        )

        task_meta = {
            "structure": struct_path.name,
            "run_dir": str(run_dir),
            "z_absorber": z_abs,
        }
        pending_tasks.append((task_meta, fut))

    async def wait_for_task(meta, parsl_future):
        try:
            await asyncio.wrap_future(parsl_future)
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
                "message": str(e),
            }

    results = await asyncio.gather(
        *(wait_for_task(meta, fut) for meta, fut in pending_tasks)
    )

    summary_log_path = output_dir / "xanes_results.jsonl"
    success_count = 0

    with open(summary_log_path, "a", encoding="utf-8") as f:
        for res in results:
            if res.get("status") == "success":
                success_count += 1
            f.write(json.dumps(res) + "\n")

    return (
        f"Ensemble execution completed. Ran {len(results)} tasks "
        f"({success_count} successful). "
        f"Detailed results appended to '{summary_log_path}'."
    )


@mcp.tool(
    name="fetch_mp_structures",
    description="Fetch optimized structures from Materials Project.",
)
def fetch_mp_structures(params: mp_query_schema):
    """Fetch structures from Materials Project and save as CIF files and pickle database."""
    from chemgraph.tools.xanes_core import (
        fetch_materials_project_data,
        _get_data_dir,
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
    from chemgraph.tools.xanes_core import (
        plot_xanes_results,
        _get_data_dir,
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
