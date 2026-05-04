"""LangChain ``@tool`` wrappers for cheminformatics functions.

Each tool delegates to the pure-Python implementation in
:mod:`chemgraph.tools.cheminformatics_core`.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.tools import tool

from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.tools.cheminformatics_core import (
    molecule_name_to_smiles_core,
    smiles_to_atomsdata_core,
    smiles_to_coordinate_file_core,
)


@tool
def molecule_name_to_smiles(name: str) -> dict:
    """Convert a molecule name to SMILES format.

    Parameters
    ----------
    name : str
        The name of the molecule to convert.

    Returns
    -------
    dict
        A JSON-serializable dict with the resolved SMILES.
    """
    smiles = molecule_name_to_smiles_core(name)
    return {"name": str(name), "smiles": smiles}


@tool
def smiles_to_atomsdata(smiles: str, randomSeed: int = 2025) -> AtomsData:
    """Convert a SMILES string to AtomsData format.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    randomSeed : int, optional
        Random seed for RDKit 3D structure generation, by default 2025.

    Returns
    -------
    AtomsData
        AtomsData object containing the molecular structure.
    """
    return smiles_to_atomsdata_core(smiles, seed=randomSeed)


@tool
def smiles_to_coordinate_file(
    smiles: str,
    output_file: str = "molecule.xyz",
    randomSeed: int = 2025,
    fmt: Literal["xyz"] = "xyz",
) -> str:
    """Convert a SMILES string to a coordinate file.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    output_file : str, optional
        Path to save the output coordinate file (currently XYZ only).
    randomSeed : int, optional
        Random seed for RDKit 3D structure generation, by default 2025.
    fmt : {"xyz"}, optional
        Output format. Only "xyz" supported for now.

    Returns
    -------
    str
        A single-line JSON string LLMs can parse.
    """
    return smiles_to_coordinate_file_core(
        smiles, output_file=output_file, seed=randomSeed, fmt=fmt
    )
