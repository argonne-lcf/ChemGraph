"""LangChain ``@tool`` wrappers for XANES/FDMNES functions.

Each tool delegates to the pure-Python implementation in
:mod:`chemgraph.tools.xanes_core`.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from chemgraph.schemas.xanes_schema import xanes_input_schema, mp_query_schema
from chemgraph.tools.xanes_core import (
    # Re-export core helpers so existing ``from xanes_tools import ...``
    # statements in MCP servers continue to work during the transition.
    write_fdmnes_input,
    get_normalized_xanes,
    extract_conv,
    _get_data_dir,
    run_xanes_core,
    fetch_materials_project_data,
    create_fdmnes_inputs,
    expand_database_results,
    plot_xanes_results,
)

# Make re-exports explicit for linters.
__all__ = [
    "write_fdmnes_input",
    "get_normalized_xanes",
    "extract_conv",
    "_get_data_dir",
    "run_xanes_core",
    "fetch_materials_project_data",
    "create_fdmnes_inputs",
    "expand_database_results",
    "plot_xanes_results",
    "run_xanes",
    "fetch_xanes_data",
    "plot_xanes_data",
]


@tool
def run_xanes(params: xanes_input_schema) -> str:
    """Run a single XANES/FDMNES calculation for one structure file.

    This tool reads the structure, generates FDMNES input files, runs FDMNES,
    and returns the result status. Requires the FDMNES_EXE environment variable.
    """
    result = run_xanes_core(params)
    if result["status"] == "success":
        return (
            f"XANES calculation completed successfully. "
            f"Output directory: {result['output_dir']}. "
            f"Found {result['n_conv_files']} convolution output(s)."
        )
    else:
        raise RuntimeError(
            f"FDMNES calculation failed in {result['output_dir']}: "
            f"{result.get('error', 'unknown error')}"
        )


@tool
def fetch_xanes_data(params: mp_query_schema) -> str:
    """Fetch optimized bulk structures from Materials Project for XANES analysis.

    Requires a Materials Project API key via the mp_api_key parameter
    or the MP_API_KEY environment variable.
    """
    data_dir = _get_data_dir()
    result = fetch_materials_project_data(params, data_dir)
    return (
        f"Fetched {result['n_structures']} structures for {params.chemsys} "
        f"into {data_dir}. "
        f"Structure files: {result['structure_files']}"
    )


@tool
def plot_xanes_data(runs_dir: str) -> str:
    """Generate normalized XANES plots for completed FDMNES calculations.

    Produces a xanes_plot.png in each run directory that contains
    FDMNES convolution output files (*_conv.txt).

    Parameters
    ----------
    runs_dir : str
        Path to the directory containing ``run_*`` subdirectories
        with FDMNES outputs.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise ValueError(f"'{runs_dir}' is not a valid directory.")

    data_dir = _get_data_dir()
    result = plot_xanes_results(data_dir, runs_path)
    if result["n_failed"] > 0:
        return (
            f"Generated {result['n_plots']} plot(s), "
            f"{result['n_failed']} failed ({result['failed']}). "
            f"Plot files: {result['plot_files']}"
        )
    return f"Generated {result['n_plots']} plot(s). Plot files: {result['plot_files']}"
