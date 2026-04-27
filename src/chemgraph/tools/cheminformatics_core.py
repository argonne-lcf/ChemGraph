"""Pure-Python cheminformatics helpers (no LangChain / MCP decorators).

Provides a single implementation for PubChem lookups and RDKit
SMILES-to-3D conversion, used by both the LangChain ``@tool`` wrappers
in :mod:`cheminformatics_tools` and the MCP wrappers in
:mod:`chemgraph.mcp.mcp_tools`.
"""

from __future__ import annotations

import os
from typing import Literal

import pubchempy as pcp

from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.tools.ase_core import _resolve_path


# ---------------------------------------------------------------------------
# SMILES → 3D coordinates (single implementation)
# ---------------------------------------------------------------------------


def smiles_to_3d(
    smiles: str, seed: int = 2025
) -> tuple[list[int], list[list[float]]]:
    """Convert a SMILES string to 3D coordinates via RDKit.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    seed : int, optional
        Random seed for reproducible 3D embedding, by default 2025.

    Returns
    -------
    tuple[list[int], list[list[float]]]
        ``(atomic_numbers, positions)`` where *positions* is a list of
        ``[x, y, z]`` lists in Angstroms.

    Raises
    ------
    ValueError
        If the SMILES string is invalid or 3D generation/optimization fails.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")

    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError("Failed to generate 3D coordinates.")
    if AllChem.UFFOptimizeMolecule(mol) != 0:
        raise ValueError("Failed to optimize 3D geometry.")

    conf = mol.GetConformer()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions = [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    return numbers, positions


# ---------------------------------------------------------------------------
# PubChem name → SMILES
# ---------------------------------------------------------------------------


def molecule_name_to_smiles_core(name: str) -> str:
    """Resolve a molecule name to its canonical SMILES via PubChem.

    Parameters
    ----------
    name : str
        Common or IUPAC molecule name.

    Returns
    -------
    str
        Canonical SMILES string.

    Raises
    ------
    ValueError
        If no PubChem match is found or the returned SMILES is empty.
    """
    if not name or not str(name).strip():
        raise ValueError("Parameter 'name' must be a non-empty string.")

    comps = pcp.get_compounds(str(name).strip(), "name")
    if not comps:
        raise ValueError(f"No PubChem compound found for name: {name!r}")

    smiles = comps[0].canonical_smiles
    if not smiles:
        raise ValueError(f"PubChem returned an empty SMILES for {name!r}.")
    return smiles


# ---------------------------------------------------------------------------
# SMILES → coordinate file
# ---------------------------------------------------------------------------


def smiles_to_coordinate_file_core(
    smiles: str,
    output_file: str = "molecule.xyz",
    seed: int = 2025,
    fmt: Literal["xyz"] = "xyz",
) -> dict:
    """Convert a SMILES string to a coordinate file on disk.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    output_file : str, optional
        Path to save the output coordinate file.
    seed : int, optional
        Random seed for RDKit 3D structure generation, by default 2025.
    fmt : {"xyz"}, optional
        Output format.  Only ``"xyz"`` is supported currently.

    Returns
    -------
    dict
        ``{"ok": True, "artifact": "coordinate_file", "path": ...,
        "smiles": ..., "natoms": ...}``

    Raises
    ------
    ValueError
        If the SMILES string is invalid or 3D generation fails.
    """
    from ase import Atoms
    from ase.io import write as ase_write

    numbers, positions = smiles_to_3d(smiles, seed=seed)
    atoms = Atoms(numbers=numbers, positions=positions)

    final_output_file = _resolve_path(output_file)
    ase_write(final_output_file, atoms)

    return {
        "ok": True,
        "artifact": "coordinate_file",
        "path": os.path.abspath(final_output_file),
        "smiles": smiles,
        "natoms": len(numbers),
    }


# ---------------------------------------------------------------------------
# SMILES → AtomsData
# ---------------------------------------------------------------------------


def smiles_to_atomsdata_core(smiles: str, seed: int = 2025) -> AtomsData:
    """Convert a SMILES string to an :class:`~chemgraph.schemas.atomsdata.AtomsData`.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule.
    seed : int, optional
        Random seed for RDKit 3D structure generation, by default 2025.

    Returns
    -------
    AtomsData
        Structure with no periodic boundary conditions.

    Raises
    ------
    ValueError
        If the SMILES string is invalid or 3D generation fails.
    """
    numbers, positions = smiles_to_3d(smiles, seed=seed)
    return AtomsData(
        numbers=numbers,
        positions=positions,
        cell=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        pbc=[False, False, False],
    )
