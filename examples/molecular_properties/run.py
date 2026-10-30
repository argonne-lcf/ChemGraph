"""Local experimental gas-phase conformer → DFT → orbital descriptor workflow."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from chemgraph.tools.molecular_core import (
    calculate_reactivity_descriptors,
    prepare_conformers,
    sha256_file,
    validate_geometry_identity,
    molecule_from_smiles,
)


def validate_manifest(data):
    from rdkit import Chem
    from chemgraph.schemas.molecular import MolecularRecord
    from chemgraph.tools.ase_core import _resolve_existing_path
    from ase.io import read
    import numpy as np

    record = MolecularRecord.model_validate(
        {k: v for k, v in data.items() if k not in {'status', 'message'}}
    )
    if (
        record.molecule_id
        != hashlib.sha256(record.isomeric_smiles.encode()).hexdigest()[:16]
    ):
        raise ValueError("Manifest molecule identity mismatch")
    if len({c.candidate_id for c in record.conformers}) != len(record.conformers):
        raise ValueError("Manifest contains duplicate conformers")
    if record.success != bool(record.conformers):
        raise ValueError("Manifest success disagrees with conformer count")
    source = molecule_from_smiles(record.isomeric_smiles, record.formal_charge)
    reference = Chem.MolFromMolBlock(record.reference_molblock, removeHs=False)
    if reference is None or Chem.MolToSmiles(
        Chem.RemoveHs(reference), isomericSmiles=True
    ) != Chem.MolToSmiles(Chem.RemoveHs(source), isomericSmiles=True):
        raise ValueError("Manifest reference disagrees with source SMILES")
    correspondence = [
        dict(
            index=a.GetIdx(),
            atomic_number=a.GetAtomicNum(),
            isotope=a.GetIsotope(),
            formal_charge=a.GetFormalCharge(),
            atom_map=a.GetAtomMapNum(),
        )
        for a in reference.GetAtoms()
    ]
    if (
        correspondence != record.atom_correspondence
        or Chem.GetFormalCharge(reference) != record.formal_charge
    ):
        raise ValueError("Manifest atom correspondence/charge mismatch")
    sdf = _resolve_existing_path(record.sdf_path)
    if sha256_file(sdf) != record.sdf_sha256:
        raise ValueError("Manifest SDF hash mismatch")
    molecules = (
        list(Chem.SDMolSupplier(sdf, removeHs=False)) if record.conformers else []
    )
    if len(molecules) != len(record.conformers):
        raise ValueError("Manifest SDF conformer count mismatch")
    for conformer, mol in zip(record.conformers, molecules):
        if conformer.conformer_id != f"{record.molecule_id}-{conformer.candidate_id}":
            raise ValueError("Manifest conformer identity mismatch")
        xyz = _resolve_existing_path(conformer.xyz_path)
        if mol is None or sha256_file(xyz) != conformer.xyz_sha256:
            raise ValueError("Manifest conformer artifact mismatch")
        if (
            Chem.MolToSmiles(mol) != Chem.MolToSmiles(reference)
            or mol.GetProp('_Name') != conformer.conformer_id
        ):
            raise ValueError("SDF identity mismatch")
        atoms = read(xyz)
        if list(atoms.numbers) != [
            a.GetAtomicNum() for a in reference.GetAtoms()
        ] or not np.allclose(
            atoms.positions, mol.GetConformer().GetPositions(), atol=5.1e-5, rtol=0
        ):
            raise ValueError("SDF/XYZ atom order or geometry mismatch")
        validate_geometry_identity(reference, atoms.positions)
    return record, reference


def run_workflow(
    smiles,
    functional,
    basis,
    output_dir="molecular_properties",
    *,
    preparation_settings=None,
):
    """Run sequential bounded calculations; return (summary, CLI exit code)."""
    from ase.io import write
    from chemgraph.schemas.ase_input import ASEInputSchema
    from chemgraph.schemas.calculators.orca_calc import OrcaCalc, DFTSettings
    from chemgraph.schemas.atomsdata import AtomsData
    from chemgraph.tools.ase_core import (
        run_ase_core,
        atomsdata_to_atoms,
        _atomic_result,
    )
    from chemgraph.runtime.context import register_artifact

    dft = DFTSettings(functional=functional, basis=basis)
    prep = prepare_conformers(smiles, output_dir, **(preparation_settings or {}))
    directory = Path(prep['run_directory'])
    summary = dict(
        schema_version=1,
        status="failure",
        preparation_manifest=prep['manifest_path'],
        preparation_sha256=sha256_file(prep['manifest_path']),
        dft_settings=dft.model_dump(),
        optimization_settings={
            "optimizer": "bfgs",
            "fmax": 0.05,
            "force_unit": "eV/angstrom",
            "steps": 200,
        },
        interpretation="Screening geometry; no minimum/stability proof or free-energy populations. Orbital descriptors are estimates.",
        timeout_guidance="Default per-evaluation/call limits are 3600/7200 s, for small/screening molecules. Operators can raise CHEMGRAPH_ORCA_CALL_TIMEOUT_S (maximum 86400 s) for larger jobs.",
        conformers=[],
        ranking=[],
    )
    try:
        manifest, reference = validate_manifest(prep)
        if not manifest.success:
            raise ValueError(prep['message'])
        for conformer in manifest.conformers:
            row = dict(
                conformer_id=conformer.conformer_id,
                molecule_id=manifest.molecule_id,
                charge=manifest.formal_charge,
                multiplicity=1,
                status="failure",
                stage="optimization",
                scientific_settings=dft.model_dump(),
                artifacts={"prepared_xyz": conformer.xyz_path},
            )
            try:
                calc = OrcaCalc(
                    charge=manifest.formal_charge, multiplicity=1, dft_settings=dft
                )
                opt_path = directory / f"{conformer.conformer_id}-opt.json"
                row['artifacts']['optimization'] = str(opt_path)
                params = ASEInputSchema(
                    input_structure_file=conformer.xyz_path,
                    output_results_file=str(opt_path),
                    driver='opt',
                    calculator=calc,
                    optimizer='bfgs',
                    fmax=0.05,
                    steps=200,
                )
                result = run_ase_core(params)
                if (
                    result.get('status') != 'success'
                    or result.get('converged') is not True
                ):
                    row['stage'] = result.get('stage', 'optimization')
                    raise ValueError(
                        result.get('message', 'Optimization failed or did not converge')
                    )
                opt_data = json.loads(opt_path.read_text())
                if (
                    opt_data.get('success') is not True
                    or opt_data.get('converged') is not True
                ):
                    raise ValueError("Saved optimization did not converge")
                saved_calc = opt_data['simulation_input']['calculator']
                if (
                    saved_calc['charge'] != manifest.formal_charge
                    or saved_calc['multiplicity'] != 1
                    or saved_calc['dft_settings'] != dft.model_dump()
                ):
                    raise ValueError("Saved optimization charge/spin/settings mismatch")
                row['stage'] = 'geometry_identity'
                final = atomsdata_to_atoms(
                    AtomsData.model_validate(opt_data['final_structure'])
                )
                if list(final.numbers) != [
                    a.GetAtomicNum() for a in reference.GetAtoms()
                ]:
                    raise ValueError("Optimized atom order changed")
                validate_geometry_identity(reference, final.positions)
                optimized_path = directory / f"{conformer.conformer_id}-optimized.xyz"
                write(optimized_path, final, format='extxyz')
                register_artifact(str(optimized_path))
                row['artifacts']['optimized_xyz'] = str(optimized_path)
                row['stage'] = 'electronic'
                electronic_path = (
                    directory / f"{conformer.conformer_id}-electronic.json"
                )
                row['artifacts']['electronic'] = str(electronic_path)
                result = run_ase_core(
                    ASEInputSchema(
                        input_structure_file=str(optimized_path),
                        output_results_file=str(electronic_path),
                        driver='electronic',
                        calculator=calc,
                    )
                )
                if result.get('status') != 'success':
                    row['stage'] = result.get('stage', 'electronic')
                    raise ValueError(
                        result.get('message', 'Electronic evaluation failed')
                    )
                data = json.loads(electronic_path.read_text())
                electronic = data['electronic_structure']
                row['warnings'] = electronic.get('warnings', [])
                electronic.update(
                    molecule_id=manifest.molecule_id,
                    conformer_id=conformer.conformer_id,
                )
                data.setdefault('provenance', {})['workflow'] = {
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in {
                        "preparation": prep['manifest_path'],
                        "optimization": opt_path,
                        "optimized_xyz": optimized_path,
                    }.items()
                }
                _atomic_result(electronic_path, data)
                row['stage'] = 'descriptors'
                descriptors = calculate_reactivity_descriptors(str(electronic_path))
                row.update(
                    status='success',
                    stage='complete',
                    energy=electronic['total_energy'],
                    energy_unit='eV',
                    descriptors=descriptors,
                    warnings=descriptors['warnings'],
                    native_settings=electronic['native_settings'],
                    evidence={
                        k: {"path": v, "sha256": sha256_file(v)}
                        for k, v in row['artifacts'].items()
                    },
                )
            except Exception as exc:
                row['reason'] = str(exc)
            summary['conformers'].append(row)
        successes = [row for row in summary['conformers'] if row['status'] == 'success']
        summary['ranking'] = [
            row['conformer_id']
            for row in sorted(
                successes, key=lambda row: (row['energy'], row['conformer_id'])
            )
        ]
        code = (
            0 if len(successes) == len(manifest.conformers) else (3 if successes else 1)
        )
        summary['status'] = {0: 'success', 3: 'partial', 1: 'failure'}[code]
    except Exception as exc:
        summary['reason'] = str(exc)
        code = 1
    summary['exit_code'] = code
    summary['summary_json'] = str(directory / 'summary.json')
    summary['summary_csv'] = str(directory / 'summary.csv')
    _atomic_result(summary['summary_json'], summary)
    fields = [
        'conformer_id',
        'molecule_id',
        'charge',
        'multiplicity',
        'status',
        'stage',
        'reason',
        'energy',
        'energy_unit',
        'functional',
        'basis',
        'ionization_energy',
        'electron_affinity',
        'chemical_potential',
        'hardness',
        'electrophilicity',
        'warnings',
        'evidence',
    ]
    with open(summary['summary_csv'], 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in summary['conformers']:
            flat = {key: row.get(key, '') for key in fields}
            flat.update(dft.model_dump())
            flat.update(
                {key: row.get('descriptors', {}).get(key, '') for key in fields[12:17]}
            )
            flat['warnings'] = json.dumps(row.get('warnings', []))
            flat['evidence'] = json.dumps(row.get('evidence', row['artifacts']))
            writer.writerow(flat)
    register_artifact(summary['summary_csv'])
    return summary, code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smiles', required=True)
    parser.add_argument('--functional', required=True)
    parser.add_argument('--basis', required=True)
    parser.add_argument('--output-dir', default='molecular_properties')
    args = parser.parse_args(argv)
    try:
        summary, code = run_workflow(
            args.smiles, args.functional, args.basis, args.output_dir
        )
        print(json.dumps(summary, indent=2))
        return code
    except Exception as exc:
        print(json.dumps({"status": "failure", "message": str(exc)}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
