import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem

from chemgraph.tools.molecular_core import (
    prepare_conformers,
    molecule_from_smiles,
    validate_geometry_identity,
)


@pytest.mark.parametrize(
    'smiles,charge',
    [
        ('O', 0),
        ('[NH4+]', 1),
        ('CN', 0),
        ('CC(=O)[O-]', -1),
        ('C[N@H+](CC)CCC', 1),
        ('F/C=C/F', 0),
        ('N[C@@H](C)C(=O)O', 0),
        ('CS(=O)(=O)N', 0),
        ('C[N+](=O)[O-]', 0),
    ],
)
def test_real_rdkit_identity_and_determinism(tmp_path, smiles, charge):
    first = prepare_conformers(smiles, str(tmp_path), candidates=3, retained=3)
    second = prepare_conformers(smiles, str(tmp_path), candidates=3, retained=3)
    assert first['success'] and first['formal_charge'] == charge
    assert first['run_directory'] != second['run_directory']
    assert first['settings']['mmff_max_iters'] == 2000
    assert [c['candidate_id'] for c in first['conformers']] == [
        c['candidate_id'] for c in second['conformers']
    ]
    for a, b in zip(first['conformers'], second['conformers']):
        assert Path(a['xyz_path']).read_bytes() == Path(b['xyz_path']).read_bytes()
    from examples.molecular_properties.run import validate_manifest

    record, _ = validate_manifest(first)
    assert record.atom_correspondence == first['atom_correspondence']
    assert json.loads(Path(first['manifest_path']).read_text())['success']


@pytest.mark.parametrize(
    'smiles', ['CC.O', '[CH3]', '[Na+]', '[H][H]', 'CC(F)Cl', 'CC=CC', 'C[NH+](CC)CCC']
)
def test_rejects_unsupported_or_unspecified_inputs_without_artifacts(tmp_path, smiles):
    with pytest.raises(ValueError):
        prepare_conformers(smiles, str(tmp_path))
    assert not list(tmp_path.iterdir())


def test_charge_settings_and_nonconvergence(tmp_path):
    with pytest.raises(ValueError, match='charge'):
        molecule_from_smiles('[NH4+]', 0)
    for settings in [
        dict(candidates=0),
        dict(candidates=1001),
        dict(retained=21),
        dict(mmff_max_iters=0),
        dict(rmsd_threshold=float('nan')),
    ]:
        with pytest.raises(ValueError):
            prepare_conformers('CN', str(tmp_path), **settings)
    result = prepare_conformers(
        'CCCCCC', str(tmp_path), candidates=2, retained=2, mmff_max_iters=1
    )
    assert not result['success']
    assert all(r['convergence_code'] != 0 for r in result['candidates'])
    assert Path(result['manifest_path']).exists()


def test_drug_sized_preparation(tmp_path):
    result = prepare_conformers(
        'CCOc1ccc(CCNC(=O)c2ccc(OCCCCOc3ccccc3)cc2)cc1',
        str(tmp_path),
        candidates=3,
        retained=3,
    )
    assert result['success']
    assert len(result['atom_correspondence']) == 63
    assert all(np.isfinite(c['mmff_energy']) for c in result['conformers'])


def test_default_settings_and_identity_change(tmp_path):
    result = prepare_conformers('CN', str(tmp_path))
    assert result['settings']['candidates'] == 20
    assert result['settings']['retained'] == 5
    assert result['rmsd_atom_set'] == 'all'
    reference = Chem.MolFromMolBlock(result['reference_molblock'], removeHs=False)
    coords = reference.GetConformer().GetPositions().copy()
    coords[0] += 10
    with pytest.raises(ValueError, match='connectivity'):
        validate_geometry_identity(reference, coords)


def test_partial_and_failed_embedding_records(tmp_path, monkeypatch):
    from rdkit.Chem import AllChem

    original = AllChem.EmbedMultipleConfs

    def partial(mol, numConfs, params):
        return original(mol, numConfs=1, params=params)

    monkeypatch.setattr(AllChem, 'EmbedMultipleConfs', partial)
    result = prepare_conformers('CN', str(tmp_path), candidates=3, retained=2)
    assert result['success']
    assert len([r for r in result['candidates'] if r['stage'] == 'embedding']) == 2

    def failure(*a, **kw):
        raise ValueError('synthetic embedding failure')

    monkeypatch.setattr(AllChem, 'EmbedMultipleConfs', failure)
    result = prepare_conformers('CN', str(tmp_path), candidates=3, retained=2)
    assert not result['success']
    assert all(
        'synthetic embedding failure' in r['reason'] for r in result['candidates']
    )
    assert Path(result['manifest_path']).exists()


def test_mirrored_defined_stereo_fails_and_manifest_tampering(tmp_path):
    from examples.molecular_properties.run import validate_manifest

    result = prepare_conformers('C[C@H](F)Cl', str(tmp_path), candidates=2, retained=2)
    reference = Chem.MolFromMolBlock(result['reference_molblock'], removeHs=False)
    coords = reference.GetConformer().GetPositions().copy()
    coords[:, 0] *= -1
    with pytest.raises(ValueError, match='stereochemistry'):
        validate_geometry_identity(reference, coords)
    for key, value in [
        ('formal_charge', 1),
        ('molecule_id', 'changed'),
        ('sdf_sha256', 'changed'),
    ]:
        with pytest.raises(ValueError):
            validate_manifest({**result, key: value})
    Path(result['conformers'][0]['xyz_path']).write_text('changed')
    with pytest.raises(ValueError, match='artifact mismatch'):
        validate_manifest(result)


def test_cli_errors_and_failed_preparation_summary(tmp_path, monkeypatch, capsys):
    from examples.molecular_properties import run
    from rdkit.Chem import AllChem

    with pytest.raises(SystemExit) as exc:
        run.main([])
    assert exc.value.code == 2
    assert (
        run.main(
            [
                '--smiles',
                'invalid',
                '--functional',
                'PBE',
                '--basis',
                'def2-SVP',
                '--output-dir',
                str(tmp_path),
            ]
        )
        == 1
    )
    monkeypatch.setattr(AllChem, 'EmbedMultipleConfs', lambda *a, **kw: [])
    assert (
        run.main(
            [
                '--smiles',
                'CN',
                '--functional',
                'PBE',
                '--basis',
                'def2-SVP',
                '--output-dir',
                str(tmp_path),
            ]
        )
        == 1
    )
    summaries = list(tmp_path.glob('*/summary.json'))
    assert len(summaries) == 1
    summary = json.loads(summaries[0].read_text())
    assert summary['exit_code'] == 1 and 'No finite' in summary['reason']
