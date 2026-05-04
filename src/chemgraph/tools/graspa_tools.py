"""LangChain ``@tool`` wrapper for gRASPA simulations.

Delegates to the pure-Python implementation in
:mod:`chemgraph.tools.graspa_core`.
"""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools.graspa_core import (
    # Re-export core helpers so existing ``from graspa_tools import ...``
    # statements (e.g. in graspa_mcp_parsl.py) continue to work.
    _read_graspa_sycl_output,
    mock_graspa,
    run_graspa_core,
)

__all__ = [
    "_read_graspa_sycl_output",
    "mock_graspa",
    "run_graspa_core",
    "run_graspa",
]


@tool
def run_graspa(graspa_input: graspa_input_schema):
    """Run a gRASPA simulation using the core engine and return the uptakes.

    This tool acts as a wrapper for the agentic workflow.
    """
    params = graspa_input_schema(
        input_structure_file=graspa_input.cif_path,
        adsorbate=graspa_input.adsorbate,
        temperature=graspa_input.temperature,
        pressure=graspa_input.pressure,
        n_cycles=graspa_input.n_cycle,
        output_result_file="raspa.log",
    )

    result = run_graspa_core(params)

    if result["status"] == "success":
        return result["uptake_in_mol_kg"]
    else:
        raise RuntimeError(f"gRASPA simulation failed for {graspa_input.mof_name}")
