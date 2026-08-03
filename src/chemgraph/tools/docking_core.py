"""Pure-Python molecular docking helpers (no LangChain / MCP decorators).

Docks a small-molecule candidate into a receptor with AutoDock Vina and returns
the predicted binding affinity and poses. Used by the LangChain ``@tool`` wrapper
in :mod:`chemgraph.tools.docking_tools`.

Heavy/optional dependencies (``vina``, ``meeko``, fpocket) are used lazily so the
core package installs and collects tests without the ``docking`` extra. A
:func:`mock_docking` helper provides deterministic output for hermetic tests.
"""

from __future__ import annotations

import os

from chemgraph.schemas.docking_schema import docking_input_schema

_BOX_PADDING = 8.0  # Angstrom added around detected/blind boxes


# ---------------------------------------------------------------------------
# Candidate resolution (SMILES / name / PubChem CID)
# ---------------------------------------------------------------------------


def resolve_candidate_smiles(candidate: str) -> str:
    """Resolve a SMILES, molecule name, or PubChem CID to a canonical SMILES.

    A valid SMILES is canonicalized; an all-digit string is treated as a PubChem
    CID; anything else is looked up by name on PubChem (reusing
    :func:`chemgraph.tools.cheminformatics_core.molecule_name_to_smiles_core`).

    Parameters
    ----------
    candidate : str
        A SMILES string, a molecule name, or a PubChem CID.

    Returns
    -------
    str
        Canonical SMILES string.
    """
    from rdkit import Chem
    from rdkit.rdBase import BlockLogs

    s = str(candidate).strip()
    with BlockLogs():  # a name/CID is not valid SMILES; suppress the expected probe error
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


def _mol_from_smiles_3d(smiles: str, seed: int = 2025):
    """Build a 3D, H-added RDKit mol from a SMILES string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, randomSeed=seed) != 0:
        raise ValueError("Failed to generate 3D coordinates.")
    # MMFF optimization is best-effort; a nonzero return just means "not converged".
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def _meeko_pdbqt(mol) -> str:
    """Prepare a molecule with Meeko and return its PDBQT string."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    setups = MoleculePreparation().prepare(mol)
    if not setups:
        raise RuntimeError("Meeko could not prepare the molecule.")
    written = PDBQTWriterLegacy.write_string(setups[0])
    pdbqt = written[0] if isinstance(written, tuple) else written
    if not pdbqt or not str(pdbqt).strip():
        raise RuntimeError("Meeko produced an empty PDBQT.")
    return pdbqt


def _prepare_ligand_pdbqt(smiles: str, out_pdbqt: str, seed: int = 2025) -> str:
    """SMILES -> 3D (RDKit) -> flexible ligand PDBQT (Meeko)."""
    with open(out_pdbqt, "w") as fh:
        fh.write(_meeko_pdbqt(_mol_from_smiles_3d(smiles, seed=seed)))
    return out_pdbqt


def _prepare_receptor_pdbqt(receptor: str, out_pdbqt: str) -> str:
    """Return a rigid-receptor PDBQT path.

    A ``.pdbqt`` path is used as-is; a SMILES/name/CID is built to 3D and written
    as a rigid receptor (Meeko PDBQT with the torsion tree stripped). A raw
    ``.pdb``/``.mol2``/``.sdf`` is rejected with guidance to pre-prepare it.
    """
    from chemgraph.tools.ase_core import _resolve_existing_path

    r = str(receptor).strip()
    if r.lower().endswith(".pdbqt"):
        path = _resolve_existing_path(r)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Receptor file not found: {r}")
        return path
    if r.lower().endswith((".pdb", ".mol2", ".sdf")):
        raise ValueError(
            f"Unsupported receptor format for {r!r}. Provide a prepared '.pdbqt', "
            "or a SMILES/name for a small-molecule receptor."
        )

    # SMILES / name / CID -> rigid small-molecule receptor
    smiles = resolve_candidate_smiles(r)
    pdbqt = _meeko_pdbqt(_mol_from_smiles_3d(smiles))
    atom_lines = [ln for ln in pdbqt.splitlines() if ln[:6].strip() in ("ATOM", "HETATM")]
    with open(out_pdbqt, "w") as fh:
        fh.write("\n".join(atom_lines) + "\n")
    return out_pdbqt


# ---------------------------------------------------------------------------
# Search box
# ---------------------------------------------------------------------------


def _heavy_coords(structure_path: str):
    """Heavy-atom coordinates (Nx3) from a PDB/PDBQT/PQR file."""
    import numpy as np

    pts = []
    with open(structure_path, errors="ignore") as fh:
        for line in fh:
            if line[:6].strip() in ("ATOM", "HETATM"):
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if elem.upper() not in ("H", "HD"):
                    pts.append(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )
    return np.array(pts)


def _box_from_points(pts, pad: float):
    return pts.mean(0).tolist(), ((pts.max(0) - pts.min(0)) + pad).tolist()


def _fpocket_box(receptor_pdbqt: str, pad: float):
    """Top-ranked fpocket pocket as (center, size), or None if unavailable."""
    import glob
    import shutil
    import subprocess
    import tempfile

    if shutil.which("fpocket") is None or shutil.which("obabel") is None:
        return None
    tmp = tempfile.mkdtemp(prefix="fpocket_")
    pdb = os.path.join(tmp, "rec.pdb")
    subprocess.run(
        f"obabel {receptor_pdbqt} -O {pdb}", shell=True, capture_output=True, check=False
    )
    subprocess.run(f"fpocket -f {pdb}", shell=True, capture_output=True, check=False)
    vert = sorted(glob.glob(os.path.join(tmp, "rec_out", "pockets", "pocket*_vert.pqr")))
    if not vert:
        return None
    pts = _heavy_coords(vert[0])
    return _box_from_points(pts, pad) if len(pts) else None


def _determine_box(receptor_pdbqt: str, params: docking_input_schema):
    """Return ``(center, box_size)`` honoring explicit override, else auto-detect."""
    if params.center is not None and params.box_size is not None:
        return list(params.center), list(params.box_size)

    order = {
        "auto": ["reference", "fpocket", "blind"],
        "reference": ["reference"],
        "fpocket": ["fpocket"],
        "blind": ["blind"],
    }[params.site_detection]

    for mode in order:
        if mode == "reference" and params.reference_ligand:
            from chemgraph.tools.ase_core import _resolve_existing_path

            pts = _heavy_coords(_resolve_existing_path(params.reference_ligand))
            if len(pts):
                return _box_from_points(pts, _BOX_PADDING)
        elif mode == "fpocket":
            res = _fpocket_box(receptor_pdbqt, _BOX_PADDING)
            if res:
                return res
        elif mode == "blind":
            return _box_from_points(_heavy_coords(receptor_pdbqt), _BOX_PADDING)

    # requested reference/fpocket was unavailable -> always-works blind fallback
    return _box_from_points(_heavy_coords(receptor_pdbqt), _BOX_PADDING)


# ---------------------------------------------------------------------------
# Mock docking (for hermetic tests)
# ---------------------------------------------------------------------------


def mock_docking(params: docking_input_schema) -> dict:
    """Return deterministic mock docking results for testing without Vina."""
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
    """Dock a candidate into a receptor with AutoDock Vina and return a result dict."""
    try:
        from vina import Vina
    except ImportError as e:
        raise ImportError(
            "AutoDock Vina is required for docking but is not installed. "
            "Install it from conda-forge:  conda install -c conda-forge vina"
        ) from e

    from chemgraph.tools.ase_core import _resolve_path

    smiles = resolve_candidate_smiles(params.candidate)
    ligand_pdbqt = _prepare_ligand_pdbqt(smiles, _resolve_path("candidate_ligand.pdbqt"))
    receptor_pdbqt = _prepare_receptor_pdbqt(
        params.receptor, _resolve_path("receptor.pdbqt")
    )
    center, box_size = _determine_box(receptor_pdbqt, params)

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
        "site_detection": params.site_detection,
        "box": {
            "center": [round(float(c), 3) for c in center],
            "size": [round(float(s), 1) for s in box_size],
        },
        "best_affinity_kcal_mol": min(scores) if scores else None,
        "n_poses": len(scores),
        "poses": [
            {"pose": i + 1, "affinity_kcal_mol": s} for i, s in enumerate(scores)
        ],
        "poses_file": os.path.abspath(poses_file),
    }
