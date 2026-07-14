"""LangChain ``@tool`` wrapper for PACMOF2 partial-charge assignment.

Delegates to the pure-Python implementation in
:mod:`chemgraph.tools.pacmof2_core`.
"""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.pacmof2_schema import pacmof2_input_schema
from chemgraph.tools.pacmof2_core import (
    # Re-export core helpers so the MCP worker can import them from here.
    _read_pacmof2_output,
    mock_pacmof2,
    run_pacmof2_core,
)

__all__ = [
    "_read_pacmof2_output",
    "mock_pacmof2",
    "run_pacmof2_core",
    "run_pacmof2",
]


@tool
def run_pacmof2(pacmof2_input: pacmof2_input_schema) -> dict:
    """Assign ML partial atomic charges to a MOF CIF using PACMOF2.

    Returns a compact summary (output CIF path, per-element mean charges,
    net/sum of charges). The full per-atom charges are written to the
    output CIF's ``_atom_site_charge`` column.

    Parameters
    ----------
    pacmof2_input : pacmof2_input_schema
        Input parameters for the charge assignment.

    Returns
    -------
    dict
        Charge summary from the core engine.
    """
    result = run_pacmof2_core(pacmof2_input)

    if result["status"] == "success":
        return result
    raise RuntimeError(
        f"PACMOF2 charge assignment failed for {pacmof2_input.input_structure_file}"
    )
