"""LangChain ``@tool`` wrappers over :mod:`chemgraph.tools.phonopy_core`."""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.phonopy_schema import PhonopyInputSchema
from chemgraph.tools.phonopy_core import run_phonopy_core
from chemgraph.schemas.calculators.mace_calc import _mace_lock


@tool
def run_phonopy(params: PhonopyInputSchema) -> dict:
    """Run Phonopy calculations using specified input parameters.

    Parameters
    ----------
    params : PhonopyInputSchema
        Input parameters for the Phonopy calculation, including the
        structure file, supercell configuration, and calculator.

    Returns
    -------
    dict
        Output containing calculation results, paths to plots, and status.

    Raises
    ------
    ValueError
        If the calculator is not supported or if the calculation fails.
    """
    calc_type = params.calculator.calculator_type.lower()
    if "mace" in calc_type:
        with _mace_lock:
            return run_phonopy_core(params)
    return run_phonopy_core(params)
