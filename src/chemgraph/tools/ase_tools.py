"""LangChain ``@tool`` wrappers over :mod:`chemgraph.tools.ase_core`.

Every public function here is a thin decorator that delegates to the
corresponding plain-Python implementation in ``ase_core.py``.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from langchain_core.tools import tool

from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.calculators.mace_calc import _mace_lock
from chemgraph.tools.ase_core import (
    _resolve_path,
    atoms_to_atomsdata,
    extract_output_json_core,
    run_ase_core,
    is_linear_molecule as _is_linear_molecule,
    get_symmetry_number as _get_symmetry_number,
)


@tool
def extract_output_json(json_file: str) -> Dict[str, Any]:
    """Load simulation results from a JSON file produced by run_ase."""
    return extract_output_json_core(json_file)


@tool
def file_to_atomsdata(fname: str) -> AtomsData:
    """Convert a structure file to AtomsData format using ASE.

    Parameters
    ----------
    fname : str
        Path to the input structure file (supports various formats like xyz, pdb, cif, etc.)

    Returns
    -------
    AtomsData
        Object containing the atomic structure information

    Raises
    ------
    FileNotFoundError
        If the input file doesn't exist
    ValueError
        If the file format is not supported or file is corrupted
    """
    from ase.io import read

    try:
        atoms = read(fname)
        return atoms_to_atomsdata(atoms)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {fname}")
    except Exception as e:
        raise ValueError(f"Failed to read structure file: {str(e)}")


@tool
def save_atomsdata_to_file(atomsdata: AtomsData, fname: str = "output.xyz") -> str:
    """Save an AtomsData object to a file using ASE.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object to save
    fname : str, optional
        Path to the output file, by default "output.xyz"

    Returns
    -------
    str
        Success message or error message

    Raises
    ------
    ValueError
        If saving the file fails
    """
    from ase.io import write
    from ase import Atoms

    try:
        atoms = Atoms(
            numbers=atomsdata.numbers,
            positions=atomsdata.positions,
            cell=atomsdata.cell,
            pbc=atomsdata.pbc,
        )
        final_fname = _resolve_path(fname)
        write(final_fname, atoms)
        return f"Successfully saved atomsdata to {os.path.abspath(final_fname)}"
    except Exception as e:
        raise ValueError(f"Failed to save atomsdata to file: {str(e)}")


@tool
def get_symmetry_number(atomsdata: AtomsData) -> int:
    """Get the rotational symmetry number of a molecule using Pymatgen.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object containing the molecular structure

    Returns
    -------
    int
        Rotational symmetry number of the molecule
    """
    return _get_symmetry_number(atomsdata)


@tool
def is_linear_molecule(atomsdata: AtomsData, tol=1e-3) -> bool:
    """Determine if a molecule is linear or not.

    Parameters
    ----------
    atomsdata : AtomsData
        AtomsData object containing the molecular structure
    tol : float, optional
        Tolerance to check for linear molecule, by default 1e-3

    Returns
    -------
    bool
        True if the molecule is linear, False otherwise
    """
    return _is_linear_molecule(atomsdata, tol)


@tool
def run_ase(params: ASEInputSchema) -> dict:
    """Run ASE calculations using specified input parameters.

    Parameters
    ----------
    params : ASEInputSchema
        Input parameters for the ASE calculation

    Returns
    -------
    dict
        Output containing calculation results and status

    Raises
    ------
    ValueError
        If the calculator is not supported or if the calculation fails
    """
    calc_type = params.calculator.calculator_type.lower()
    if "mace" in calc_type:
        with _mace_lock:
            return run_ase_core(params)
    return run_ase_core(params)
