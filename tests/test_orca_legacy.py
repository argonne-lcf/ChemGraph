"""Legacy ORCA compatibility in an isolated configured interpreter."""

import os
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

if os.environ.get('CHEMGRAPH_ORCA_CHILD') != '1':
    pytest.skip('Isolated configured-ORCA child only', allow_module_level=True)

from ase import Atoms
from ase.io import write
from ase.calculators.orca import OrcaProfile
from chemgraph.schemas.ase_input import ASEInputSchema, ase_input_schema_ensemble
from chemgraph.schemas.calculators.orca_calc import OrcaCalc
from chemgraph.tools import ase_core
from chemgraph.tools.ase_tools import run_ase


@pytest.fixture
def structure(tmp_path):
    path = tmp_path / 'water.xyz'
    write(path, Atoms('OH2', positions=[[0, 0, 0], [0, 0, 0.96], [0.93, 0, -0.24]]))
    return path


def test_legacy_profile_schema_serialization():
    for model in (
        OrcaCalc,
        ASEInputSchema,
        ase_input_schema_ensemble,
        run_ase.args_schema,
    ):
        schema = model.model_json_schema()
        properties = (
            schema['properties']
            if model is OrcaCalc
            else schema['$defs']['OrcaCalc']['properties']
        )
        assert 'profile' not in properties
    profile = OrcaProfile(command='false')
    config = OrcaCalc(profile=profile)
    assert 'profile' not in config.model_dump()
    assert config.get_calculator().profile is profile
    assert ase_core.load_calculator(config)[0].profile is profile


@pytest.mark.parametrize('charge,symbols', [(0, 'OH2'), (1, 'NH4'), (-1, 'C2H3O2')])
def test_legacy_actual_writer_bytes_preserved(tmp_path, charge, symbols):
    from ase.calculators.orca import ORCA

    atoms = Atoms(symbols)
    old, new = tmp_path / 'old', tmp_path / 'new'
    old.mkdir()
    new.mkdir()
    profile = OrcaProfile(command='false')
    before = ORCA(
        profile=profile,
        directory=old,
        charge=charge,
        mult=1,
        orcasimpleinput='B3LYP def2-TZVP',
        orcablocks='%pal nprocs 1 end',
    )
    after = OrcaCalc(
        profile=profile, directory=str(new), charge=charge
    ).get_calculator()
    for calc in (before, after):
        calc.template.write_input(
            profile, calc.directory, atoms, calc.parameters, ['energy']
        )
    assert (old / 'orca.inp').read_bytes() == (new / 'orca.inp').read_bytes()


def test_legacy_concurrent_directory_isolation_and_explicit_compatibility(
    structure, tmp_path, monkeypatch
):
    from ase.calculators.orca import OrcaTemplate

    def execute(self, directory, profile):
        (directory / 'orca.out').write_text('native test artifact')

    monkeypatch.setattr(OrcaTemplate, 'execute', execute)
    monkeypatch.setattr(OrcaTemplate, 'read_results', lambda *a: {'energy': -1.0})

    def run(i, directory='.'):
        return ase_core.run_ase_core(
            ASEInputSchema(
                input_structure_file=str(structure),
                output_results_file=str(tmp_path / f'legacy-{i}.json'),
                driver='energy',
                calculator=OrcaCalc(directory=directory),
            )
        )

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(2)))
    assert all(r['status'] == 'success' for r in results)
    saved = [json.loads((tmp_path / f'legacy-{i}.json').read_text()) for i in range(2)]
    assert saved[0]['working_directory'] != saved[1]['working_directory']
    assert all(any(p.endswith('orca.inp') for p in s['artifacts']) for s in saved)
    explicit = tmp_path / 'explicit'
    explicit.mkdir()
    assert run(2, str(explicit))['status'] == 'success'
    assert (explicit / 'orca.out').exists()
