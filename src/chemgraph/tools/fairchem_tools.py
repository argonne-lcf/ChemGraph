"""FairChem/UMA helpers.

``run_fairchem_core`` converts the FairChem-specific input schema to
:class:`ASEInputSchema` and delegates to
:func:`chemgraph.tools.ase_core.run_ase_core`, mirroring
:func:`chemgraph.tools.parsl_tools.run_mace_core`.
"""

from __future__ import annotations

import logging

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.fairchem_schema import (
    fairchem_input_schema,
    fairchem_output_schema,
)
from chemgraph.tools.ase_core import run_ase_core

# Re-export schemas so existing ``from chemgraph.tools.fairchem_tools import …``
# statements continue to work.
__all__ = [
    "fairchem_input_schema",
    "fairchem_output_schema",
    "run_fairchem_core",
    "extract_output_json",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core execution — delegates to the unified implementation
# ---------------------------------------------------------------------------


def _fairchem_input_to_ase_input(params: fairchem_input_schema) -> ASEInputSchema:
    """Convert a FairChem-specific input schema to a generic ASEInputSchema.

    Parameters
    ----------
    params : fairchem_input_schema
        FairChem-specific tool input.

    Returns
    -------
    ASEInputSchema
        Equivalent generic ASE input carrying a ``FAIRChem`` calculator.
    """
    return ASEInputSchema(
        input_structure_file=params.input_structure_file,
        output_results_file=params.output_result_file,
        driver=params.driver,
        optimizer=params.optimizer,
        calculator={
            "calculator_type": "FAIRChem",
            "model_name": params.model_name,
            "task_name": params.task_name,
            "device": params.device,
            "charge": params.charge,
            "multiplicity": params.multiplicity,
            "inference_settings": params.inference_settings,
            "seed": params.seed,
        },
        fmax=params.fmax,
        steps=params.steps,
        temperature=params.temperature,
        pressure=params.pressure,
    )


def run_fairchem_core(params: fairchem_input_schema) -> dict:
    """Run a single FairChem/UMA calculation.

    Converts the FairChem-specific schema to :class:`ASEInputSchema` and
    delegates to :func:`chemgraph.tools.ase_core.run_ase_core`.

    Parameters
    ----------
    params : fairchem_input_schema
        FairChem-specific input parameters.

    Returns
    -------
    dict
        Simulation result payload.
    """
    ase_params = _fairchem_input_to_ase_input(params)
    return run_ase_core(ase_params)


def extract_output_json(json_file: str) -> dict:
    """Load simulation results from a JSON file produced by run_fairchem."""
    import json

    try:
        with open(json_file, "r") as f:
            ret = json.load(f)
    except Exception:
        ret = {}
    return ret
