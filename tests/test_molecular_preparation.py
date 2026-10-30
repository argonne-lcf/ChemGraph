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
