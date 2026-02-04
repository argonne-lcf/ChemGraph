import os
from typing import Literal

import pubchempy
from langchain_core.tools import tool
from ase.io import write as ase_write
from ase import Atoms

from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.tools.mcp_helper import _resolve_path


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

    Raises
    ------
    IndexError
        If the molecule name is not found in PubChem.
    """
    smiles = pubchempy.get_compounds(str(name), "name")[0].connectivity_smiles
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

    Raises
    ------
    ValueError
        If the SMILES string is invalid or if 3D structure generation fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Generate the molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Add hydrogens and optimize 3D structure
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=randomSeed) != 0:
        raise ValueError("Failed to generate 3D coordinates.")
    if AllChem.UFFOptimizeMolecule(mol) != 0:
        raise ValueError("Failed to optimize 3D geometry.")
    # Extract atomic information
    conf = mol.GetConformer()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

    # Create AtomsData object
    atoms_data = AtomsData(
        numbers=numbers,
        positions=positions,
        cell=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        pbc=[False, False, False],  # No periodic boundary conditions
    )
    return atoms_data


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
        A single-line JSON string LLMs can parse, e.g.
        {"ok": true, "artifact": "coordinate_file", "format": "xyz", "path": "...", "smiles": "...", "natoms": 12}

    Raises
    ------
    ValueError
        If the SMILES string is invalid or if 3D structure generation fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Generate the molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    # Add hydrogens and optimize 3D structure
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=randomSeed) != 0:
        raise ValueError("Failed to generate 3D coordinates.")
    if AllChem.UFFOptimizeMolecule(mol) != 0:
        raise ValueError("Failed to optimize 3D geometry.")
    # Extract atomic information
    conf = mol.GetConformer()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]

    # Create Atoms object
    atoms = Atoms(numbers=numbers, positions=positions)

    final_output_file = _resolve_path(output_file)
    ase_write(
        final_output_file,
        atoms,
    )

    # Return dict for LLM/tool chaining
    return {
        "ok": True,
        "artifact": "coordinate_file",
        "path": os.path.abspath(final_output_file),
        "smiles": smiles,
        "natoms": len(numbers),
    }
