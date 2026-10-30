"""Deterministic conformer preparation and conservative molecular identity checks."""

import hashlib
import math
from pathlib import Path
import tempfile

from chemgraph.schemas.molecular import (
    ConformerSettings,
    MolecularRecord,
    SUPPORTED_ELEMENTS,
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_unscoped():
    from chemgraph.runtime.context import current_context

    if current_context() is not None:
        raise PermissionError(
            "Molecular property tools support local unscoped execution only"
        )


def molecule_from_smiles(smiles, charge=None):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Supply one connected, valid isomeric SMILES")
    if not any(a.GetAtomicNum() > 1 for a in mol.GetAtoms()):
        raise ValueError("At least one heavy atom is required")
    if any(
        a.GetAtomicNum() not in SUPPORTED_ELEMENTS or a.GetNumRadicalElectrons()
        for a in mol.GetAtoms()
    ):
        raise ValueError(
            "Only closed-shell H/C/N/O/F/P/S/Cl/Br molecules are supported"
        )
    stereo = Chem.FindPotentialStereo(mol)
    for item in stereo:
        if str(item.type) not in {"Atom_Tetrahedral", "Bond_Double"}:
            raise ValueError(
                "Axial/atropisomer or other non-tetrahedral stereochemistry is unsupported"
            )
        if str(item.specified) != "Specified":
            raise ValueError(
                "Specify every tetrahedral and E/Z configuration, including stereogenic protonated nitrogen"
            )
    if any(
        str(b.GetStereo()) in {"STEREOATROPCW", "STEREOATROPCCW"}
        for b in mol.GetBonds()
    ):
        raise ValueError("Atropisomer workflows are unsupported")
    formal_charge = Chem.GetFormalCharge(mol)
    if charge is not None and charge != formal_charge:
        raise ValueError("Supplied charge disagrees with SMILES formal charge")
    mol = Chem.AddHs(mol)
    if (sum(a.GetAtomicNum() for a in mol.GetAtoms()) - formal_charge) % 2:
        raise ValueError(
            "Odd electron count is incompatible with a closed-shell singlet"
        )
    if not AllChem.MMFFHasAllMoleculeParams(mol):
        raise ValueError("Molecule is unsupported by MMFF94s")
    return mol


def validate_geometry_identity(reference, positions):
    """Compare mapped adjacency and specified stereo without guessing bond orders."""
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    coords = np.asarray(positions, dtype=float)
    if coords.shape != (reference.GetNumAtoms(), 3) or not np.isfinite(coords).all():
        raise ValueError("Invalid geometry or atom count")
    bare = Chem.RWMol()
    for atom in reference.GetAtoms():
        bare.AddAtom(Chem.Atom(atom.GetAtomicNum()))
    conf = Chem.Conformer(len(coords))
    for idx, xyz in enumerate(coords):
        conf.SetAtomPosition(idx, xyz)
    bare.AddConformer(conf)
    rdDetermineBonds.DetermineConnectivity(bare, useHueckel=False, useVdw=False)

    def edges(mol):
        return {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in mol.GetBonds()
        }

    if edges(bare) != edges(reference):
        raise ValueError("Optimized geometry changed mapped connectivity (including H)")
    final = Chem.Mol(reference)
    final.RemoveAllConformers()
    final.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(final, replaceExistingTags=True)
    Chem.AssignStereochemistry(final, cleanIt=True, force=True)
    expected = Chem.Mol(reference)
    Chem.AssignStereochemistry(expected, cleanIt=True, force=True)
    for a, b in zip(expected.GetAtoms(), final.GetAtoms()):
        if a.HasProp('_CIPCode') and (
            not b.HasProp('_CIPCode') or a.GetProp('_CIPCode') != b.GetProp('_CIPCode')
        ):
            raise ValueError(
                "Defined tetrahedral stereochemistry changed or became ambiguous"
            )
    for a, b in zip(expected.GetBonds(), final.GetBonds()):
        if (
            str(a.GetStereo()) in {"STEREOE", "STEREOZ", "STEREOCIS", "STEREOTRANS"}
            and a.GetStereo() != b.GetStereo()
        ):
            raise ValueError(
                "Defined double-bond stereochemistry changed or became ambiguous"
            )
    return final


def prepare_conformers(
    smiles: str,
    output_dir: str = "molecular_properties",
    charge: int | None = None,
    candidates: int = 20,
    retained: int = 5,
    seed: int = 2025,
    rmsd_threshold: float = 0.5,
    mmff_max_iters: int = 2000,
) -> dict:
    """Prepare converged MMFF94s conformers; preserve every candidate disposition."""
    require_unscoped()
    from rdkit import Chem, rdBase
    from rdkit.Chem import AllChem, rdMolAlign
    from chemgraph.tools.ase_core import _resolve_path
    from chemgraph.runtime.context import register_artifact

    settings = ConformerSettings(
        candidates=candidates,
        retained=retained,
        seed=seed,
        rmsd_threshold=rmsd_threshold,
        mmff_max_iters=mmff_max_iters,
    )
    if retained > candidates:
        raise ValueError("retained must not exceed candidates")
    mol = molecule_from_smiles(smiles, charge)
    canonical = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    parent = Path(_resolve_path(output_dir)).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="conformers-", dir=parent))
    etkdg = AllChem.ETKDGv3()
    etkdg.randomSeed = seed
    etkdg.numThreads = 1
    etkdg.pruneRmsThresh = -1
    embedding_error = None
    try:
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=candidates, params=etkdg))
    except Exception as exc:
        ids = []
        embedding_error = str(exc)
    dispositions = [
        {
            "candidate_id": i,
            "stage": "embedding",
            "status": "failure",
            "reason": embedding_error or "Embedding did not produce a conformer",
        }
        for i in range(len(ids), candidates)
    ]
    optimized = []
    properties = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    for idx in ids:
        row = {
            "candidate_id": idx,
            "mmff_max_iters": mmff_max_iters,
            "stage": "mmff",
            "status": "failure",
        }
        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=idx)
            code = ff.Minimize(maxIts=mmff_max_iters)
            energy = ff.CalcEnergy()
            row.update(
                convergence_code=code,
                mmff_energy=energy if math.isfinite(energy) else None,
            )
            if code == 0 and math.isfinite(energy):
                row['stage'] = 'mmff_identity'
                validate_geometry_identity(mol, mol.GetConformer(idx).GetPositions())
                optimized.append((energy, idx))
                row.update(status='converged', stage='mmff')
            else:
                row['reason'] = 'MMFF failed to converge to a finite energy'
        except Exception as exc:
            row['reason'] = str(exc)
        dispositions.append(row)
    atom_set = (
        "heavy" if sum(a.GetAtomicNum() > 1 for a in mol.GetAtoms()) > 2 else "all"
    )
    chosen = []
    for energy, idx in sorted(optimized):
        duplicate = False
        for _, other in chosen:
            probe, target = Chem.Mol(mol), Chem.Mol(mol)
            if atom_set == "heavy":
                probe, target = Chem.RemoveHs(probe), Chem.RemoveHs(target)
            if (
                rdMolAlign.GetBestRMS(probe, target, prbId=idx, refId=other)
                < rmsd_threshold
            ):
                duplicate = True
                break
        disposition = next(row for row in dispositions if row['candidate_id'] == idx)
        disposition['status'] = (
            'duplicate'
            if duplicate
            else ('retained' if len(chosen) < retained else 'retained_limit')
        )
        if not duplicate and len(chosen) < retained:
            chosen.append((energy, idx))
    molecule_id = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    sdf_path = directory / "conformers.sdf"
    records = []
    with Chem.SDWriter(str(sdf_path)) as writer:
        for energy, idx in chosen:
            conformer_id = f"{molecule_id}-{idx}"
            mol.SetProp('_Name', conformer_id)
            mol.SetProp('formal_charge', str(Chem.GetFormalCharge(mol)))
            writer.write(mol, confId=idx)
            xyz = directory / f"{conformer_id}.xyz"
            xyz.write_text(Chem.MolToXYZBlock(mol, confId=idx, precision=12))
            records.append(
                dict(
                    conformer_id=conformer_id,
                    candidate_id=idx,
                    mmff_energy=energy,
                    xyz_path=str(xyz),
                    xyz_sha256=sha256_file(xyz),
                )
            )
    manifest = directory / "manifest.json"
    artifacts = [str(sdf_path), *(r['xyz_path'] for r in records), str(manifest)]
    record = MolecularRecord(
        success=bool(records),
        molecule_id=molecule_id,
        isomeric_smiles=canonical,
        formal_charge=Chem.GetFormalCharge(mol),
        atom_correspondence=[
            dict(
                index=a.GetIdx(),
                atomic_number=a.GetAtomicNum(),
                isotope=a.GetIsotope(),
                formal_charge=a.GetFormalCharge(),
                atom_map=a.GetAtomMapNum(),
            )
            for a in mol.GetAtoms()
        ],
        reference_molblock=Chem.MolToMolBlock(mol),
        rdkit_version=rdBase.rdkitVersion,
        settings=settings,
        rmsd_atom_set=atom_set,
        conformers=records,
        candidates=dispositions,
        artifacts=artifacts,
        sdf_path=str(sdf_path),
        sdf_sha256=sha256_file(sdf_path),
        run_directory=str(directory),
        manifest_path=str(manifest),
    )
    manifest.write_text(record.model_dump_json(indent=2))
    for path in artifacts:
        register_artifact(path)
    return {
        "status": "success" if records else "failure",
        **record.model_dump(),
        "message": "Conformers prepared"
        if records
        else "No finite, converged, identity-preserving conformers survived",
    }
