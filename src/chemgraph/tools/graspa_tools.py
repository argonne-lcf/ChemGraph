"""LangChain ``@tool`` wrapper for gRASPA simulations.

Delegates to the pure-Python implementation in
:mod:`chemgraph.tools.graspa_core`.
"""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools.graspa_core import run_graspa_core

__all__ = ["run_graspa"]


@tool
def run_graspa(graspa_input: graspa_input_schema):
    """Run single-component H2O, CO2, or N2 adsorption with gRASPA-SYCL.

    This tool acts as a wrapper for the agentic workflow.

    Parameters
    ----------
    graspa_input : graspa_input_schema
        Legacy gRASPA tool input.

    Returns
    -------
    float
        Uptake in mol/kg from the core gRASPA result.
    """
    result = run_graspa_core(graspa_input)

    if result["status"] == "success":
        return result["uptake_in_mol_kg"]
    else:
        raise RuntimeError(
            f"gRASPA failed for {graspa_input.input_structure_file}: "
            f"{result.get('message', 'unknown error')}; "
            f"stdout={result.get('stdout_path')}, stderr={result.get('stderr_path')}"
        )
