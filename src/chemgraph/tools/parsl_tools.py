"""Parsl-oriented MACE helpers.

``run_mace_core`` converts the MACE-specific input schema to
:class:`ASEInputSchema` and delegates to :func:`chemgraph.tools.ase_core.run_ase_core`.
"""

from __future__ import annotations

import logging

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.mace_parsl_schema import (
    mace_input_schema,
    mace_output_schema,
)
from chemgraph.tools.ase_core import run_ase_core

# Re-export schemas so existing ``from chemgraph.tools.parsl_tools import …``
# statements continue to work.
__all__ = [
    "mace_input_schema",
    "mace_output_schema",
    "run_mace_core",
    "extract_output_json",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core execution — delegates to the unified implementation
# ---------------------------------------------------------------------------


def _mace_input_to_ase_input(params: mace_input_schema) -> ASEInputSchema:
    """Convert a MACE-specific input schema to a generic ASEInputSchema.

    Parameters
    ----------
    params : mace_input_schema
        MACE-specific tool input.

    Returns
    -------
    ASEInputSchema
        Equivalent generic ASE input.
    """
    return ASEInputSchema(
        input_structure_file=params.input_structure_file,
        output_results_file=params.output_result_file,
        driver=params.driver,
        optimizer=params.optimizer,
        calculator={
            "calculator_type": "mace_mp",
            "model": params.model,
            "device": params.device,
        },
        fmax=params.fmax,
        steps=params.steps,
        temperature=params.temperature,
        pressure=params.pressure,
    )


def run_mace_core(params: mace_input_schema) -> dict:
    """Run a single MACE calculation.

    Converts the MACE-specific schema to :class:`ASEInputSchema` and
    delegates to :func:`chemgraph.tools.ase_core.run_ase_core`.

    Parameters
    ----------
    params : mace_input_schema
        MACE-specific input parameters.

    Returns
    -------
    dict
        Simulation result payload.
    """
    try:
        ase_params = _mace_input_to_ase_input(params)
        return run_ase_core(ase_params)
    except Exception as e:
        print(f"Running ase failed with error:{e}")
        return None


def extract_output_json(json_file: str) -> dict:
    """Load simulation results from a JSON file produced by run_ase."""
    import json

    with open(json_file, "r") as f:
        return json.load(f)
