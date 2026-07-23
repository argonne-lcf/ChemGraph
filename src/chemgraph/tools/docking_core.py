"""Pure-Python molecular docking helpers (no LangChain / MCP decorators).

Docks a small-molecule candidate into a receptor with AutoDock Vina and returns
the predicted binding affinity and poses. Used by the LangChain ``@tool`` wrapper
in :mod:`chemgraph.tools.docking_tools`.

Heavy/optional dependencies (``vina``, ``meeko``) are imported lazily inside the
functions that need them, so the core package installs and collects tests without
the ``docking`` extra. A :func:`mock_docking` helper provides deterministic output
for hermetic tests.
"""

from __future__ import annotations

import os
from pathlib import Path

from chemgraph.schemas.docking_schema import docking_input_schema

# Bundled, ready-to-dock vancomycin receptor and its known binding-site box
# (validated by redocking the native D-Ala-D-Ala ligand from PDB 1FVM).
_FILES = Path(__file__).parent / "files" / "docking"
_VANCOMYCIN_RECEPTOR = _FILES / "vancomycin_receptor.pdbqt"
_VANCOMYCIN_CENTER = [-3.436, 5.510, 22.100]
_VANCOMYCIN_SIZE = [18.0, 16.0, 22.0]


# ---------------------------------------------------------------------------
# Candidate resolution (SMILES / name / PubChem CID)
# ---------------------------------------------------------------------------


def resolve_candidate_smiles(candidate: str) -> str:
    """Resolve a SMILES, molecule name, or PubChem CID to a canonical SMILES.

    A valid SMILES is canonicalized and returned; an all-digit string is treated
    as a PubChem CID; anything else is looked up by name on PubChem (reusing
    :func:`chemgraph.tools.cheminformatics_core.molecule_name_to_smiles_core`).

    Parameters
    ----------
    candidate : str
        A SMILES string, a molecule name, or a PubChem CID.

    Returns
    -------
    str
        Canonical SMILES string.

    Raises
    ------
    ValueError
        If the candidate cannot be resolved.
    """
    from rdkit import Chem
    from rdkit.rdBase import BlockLogs

    s = str(candidate).strip()
    # A name or CID is not valid SMILES; suppress the expected parse-error log
    # from this probe (scoped, so other RDKit warnings are unaffected).
    with BlockLogs():
        mol = Chem.MolFromSmiles(s)
    if mol is not None:
        return Chem.MolToSmiles(mol)

    if s.isdigit():
        import pubchempy as pcp

        comps = pcp.get_compounds(s, "cid")
        if comps and comps[0].canonical_smiles:
            return comps[0].canonical_smiles

    from chemgraph.tools.cheminformatics_core import molecule_name_to_smiles_core

    return molecule_name_to_smiles_core(s)


# ---------------------------------------------------------------------------
# Ligand / receptor preparation
# ---------------------------------------------------------------------------


def _prepare_ligand_pdbqt(smiles: str, out_pdbqt: str, seed: int = 2025) -> str:
    """Build a 3D structure from SMILES (RDKit) and write a docking-ready PDBQT (Meeko)."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError("Failed to generate 3D coordinates for the candidate.")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass

    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise RuntimeError("Meeko could not prepare the candidate ligand.")
    written = PDBQTWriterLegacy.write_string(setups[0])
    pdbqt = written[0] if isinstance(written, tuple) else written
    if not pdbqt or not str(pdbqt).strip():
        raise RuntimeError("Meeko produced an empty ligand PDBQT.")
    with open(out_pdbqt, "w") as fh:
        fh.write(pdbqt)
    return out_pdbqt


def _resolve_receptor(params: docking_input_schema):
    """Return ``(receptor_pdbqt, center, box_size)`` for the requested receptor."""
    r = str(params.receptor).strip()
    if r.lower() in ("vancomycin", "1fvm"):
        return str(_VANCOMYCIN_RECEPTOR), _VANCOMYCIN_CENTER, _VANCOMYCIN_SIZE

    from chemgraph.tools.ase_core import _resolve_existing_path

    receptor_pdbqt = _resolve_existing_path(r)
    if not os.path.exists(receptor_pdbqt):
        raise FileNotFoundError(f"Receptor file not found: {r}")
    if params.center is None or params.box_size is None:
        raise ValueError(
            "A custom receptor requires both 'center' and 'box_size'."
        )
    return receptor_pdbqt, list(params.center), list(params.box_size)


# ---------------------------------------------------------------------------
# Mock docking (for hermetic tests)
# ---------------------------------------------------------------------------


def mock_docking(params: docking_input_schema) -> dict:
    """Return deterministic mock docking results for testing without Vina.

    Parameters
    ----------
    params : docking_input_schema
        Docking input; only ``candidate``, ``receptor`` and ``n_poses`` are used.

    Returns
    -------
    dict
        A result dict with the same shape as :func:`run_docking_core`.
    """
    scores = [round(-5.0 + 0.3 * i, 2) for i in range(params.n_poses)]
    return {
        "candidate": {"input": params.candidate, "smiles": params.candidate},
        "receptor": params.receptor,
        "engine": "mock",
        "best_affinity_kcal_mol": scores[0] if scores else None,
        "n_poses": len(scores),
        "poses": [
            {"pose": i + 1, "affinity_kcal_mol": s} for i, s in enumerate(scores)
        ],
        "poses_file": None,
    }


# ---------------------------------------------------------------------------
# Core docking runner
# ---------------------------------------------------------------------------


def run_docking_core(params: docking_input_schema) -> dict:
    """Dock a candidate into a receptor with AutoDock Vina.

    Parameters
    ----------
    params : docking_input_schema
        Candidate, receptor, number of poses, optional box, and exhaustiveness.

    Returns
    -------
    dict
        Result including the resolved candidate SMILES, receptor, engine, best
        binding affinity in kcal/mol (more negative = stronger), a per-pose list,
        and the path to the written poses PDBQT.
    """
    try:
        from vina import Vina
    except ImportError as e:
        raise ImportError(
            "AutoDock Vina is required for docking but is not installed. "
            "Install it from conda-forge:  conda install -c conda-forge vina"
        ) from e

    from chemgraph.tools.ase_core import _resolve_path

    smiles = resolve_candidate_smiles(params.candidate)
    ligand_pdbqt = _resolve_path("candidate_ligand.pdbqt")
    _prepare_ligand_pdbqt(smiles, ligand_pdbqt)

    receptor_pdbqt, center, box_size = _resolve_receptor(params)

    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    v.compute_vina_maps(
        center=[float(c) for c in center], box_size=[float(s) for s in box_size]
    )
    v.dock(exhaustiveness=params.exhaustiveness, n_poses=params.n_poses)

    poses_file = _resolve_path("candidate_poses.pdbqt")
    v.write_poses(poses_file, n_poses=params.n_poses, overwrite=True)
    scores = [round(float(e[0]), 2) for e in v.energies(n_poses=params.n_poses)]

    return {
        "candidate": {"input": params.candidate, "smiles": smiles},
        "receptor": params.receptor,
        "engine": "vina",
        "best_affinity_kcal_mol": min(scores) if scores else None,
        "n_poses": len(scores),
        "poses": [
            {"pose": i + 1, "affinity_kcal_mol": s} for i, s in enumerate(scores)
        ],
        "poses_file": os.path.abspath(poses_file),
    }
