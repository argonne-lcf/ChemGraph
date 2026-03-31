from pathlib import Path

from mcp.server.fastmcp import FastMCP

from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.schemas.xanes_schema import xanes_input_schema, mp_query_schema

# Start MCP server
mcp = FastMCP(
    name="ChemGraph XANES Tools",
    instructions="""
        You expose tools for running XANES/FDMNES simulations.
        The available tools are:
        1. run_xanes_single: run a single FDMNES calculation for one structure.
        2. fetch_mp_structures: fetch optimized structures from Materials Project.
        3. plot_xanes: generate normalized XANES plots for completed calculations.

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they occur.
        - Keep responses compact -- full results are in the output directories.
        - When returning paths, use absolute paths.
        - Energies are in eV.
        - The FDMNES executable path is read from the FDMNES_EXE environment variable.
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


@mcp.tool(
    name="fetch_mp_structures",
    description="Fetch optimized structures from Materials Project.",
)
def fetch_mp_structures(params: mp_query_schema):
    """Fetch structures from Materials Project and save as CIF files and pickle database."""
    from chemgraph.tools.xanes_tools import (
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
    from chemgraph.tools.xanes_tools import (
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
