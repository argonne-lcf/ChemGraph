"""Run only in a configured child; never globally enable ORCA in ordinary tests."""

import json
import os
from unittest.mock import Mock

import pytest

if os.environ.get('CHEMGRAPH_ORCA_CHILD') != '1':
    pytest.skip('Isolated configured-ORCA child only', allow_module_level=True)

from ase import Atoms
from ase.calculators.orca import OrcaProfile
from ase.io import read, write
from chemgraph.schemas.ase_input import (
    ASEInputSchema,
    ASEOutputSchema,
    ase_input_schema_ensemble,
)
from chemgraph.schemas.calculators.orca_calc import OrcaCalc
from chemgraph.tools import ase_core, orca_execution
from chemgraph.tools.ase_tools import run_ase
from tests.orca_helpers import synthetic_output, synthetic_engrad


@pytest.fixture
def structure(tmp_path):
    path = tmp_path / 'water.xyz'
    write(path, Atoms('OH2', positions=[[0, 0, 0], [0, 0, 0.96], [0.93, 0, -0.24]]))
    return path


@pytest.fixture
def fake_engine(monkeypatch):
    monkeypatch.setattr(orca_execution, 'identify_orca', lambda *a: '6.0.0')

    def execute(self, directory, profile):
        atoms = self.atoms
        active = self.active_directory
        (active / 'orca.out').write_text(
            synthetic_output(
                atoms,
                charge=self.config.charge,
                functional=self.config.dft_settings.functional,
                basis=self.config.dft_settings.basis,
                gradient=self.driver == 'opt',
            )
        )
        import re

        native_output = (active / 'orca.out').read_text()
        echoed = '\n'.join(
            f'| {i}> {line}'
            for i, line in enumerate((active / 'orca.inp').read_text().splitlines(), 1)
        )
        (active / 'orca.out').write_text(
            re.sub(r'(?m)^\| 1>.*$', lambda _: echoed, native_output)
        )
        if self.driver == 'opt':
            (active / 'orca.engrad').write_text(synthetic_engrad(atoms))

    monkeypatch.setattr(orca_execution.StructuredOrcaTemplate, 'execute', execute)
    return execute


def params(structure, output, driver='electronic', **kwargs):
    return ASEInputSchema(
        input_structure_file=str(structure),
        output_results_file=str(output),
        driver=driver,
        calculator=OrcaCalc(
            charge=0,
            multiplicity=1,
            dft_settings={'functional': 'pbe', 'basis': 'DEF2-svp'},
        ),
        **kwargs,
    )


def test_schema_visibility_profile_exclusion_roundtrips(structure, tmp_path):
    for model in (ASEInputSchema, ase_input_schema_ensemble, run_ase.args_schema):
        schema = model.model_json_schema()
        assert 'OrcaCalc' in schema['$defs']
        assert 'profile' not in schema['$defs']['OrcaCalc']['properties']
        assert 'dft_settings' in schema['$defs']['OrcaCalc']['properties']
        assert 'command' not in schema['$defs']['OrcaCalc']['properties']
    profile = OrcaProfile(command=os.environ['ASE_ORCA_COMMAND'])
    legacy = OrcaCalc(profile=profile)
    p = ASEInputSchema(input_structure_file=str(structure), calculator=legacy)
    assert 'profile' not in json.loads(p.model_dump_json())['calculator']
    assert ase_core.load_calculator(legacy)[0].profile is profile
    assert legacy.get_calculator().profile is profile
    output = ASEOutputSchema(
        input_structure_file=str(structure),
        final_structure=ase_core.atoms_to_atomsdata(read(structure)),
        simulation_input=p,
    )
    assert (
        'profile'
        not in json.loads(output.model_dump_json())['simulation_input']['calculator']
    )
    p = params(structure, tmp_path / 'out.json')
    for clone in [
        ASEInputSchema.model_validate(p.model_dump()),
        ASEInputSchema.model_validate_json(p.model_dump_json()),
    ]:
        assert clone.calculator.dft_settings.functional == 'PBE'
        assert clone.calculator.orcasimpleinput is None
        assert clone.calculator.orcablocks is None
    ensemble = ase_input_schema_ensemble(calculator=p.calculator, driver='electronic')
    assert (
        ase_input_schema_ensemble.model_validate_json(
            ensemble.model_dump_json()
        ).calculator
        == p.calculator
    )


@pytest.mark.parametrize(
    'extra',
    [
        dict(orcasimpleinput='B3LYP def2-TZVP'),
        dict(orcablocks='%pal nprocs 1 end'),
        dict(directory='/tmp'),
        dict(profile=OrcaProfile(command='false')),
        dict(command='orca'),
        dict(multiplicity=3),
    ],
)
def test_structured_conflicts(extra):
    with pytest.raises(ValueError):
        OrcaCalc(
            **{
                'charge': 0,
                'multiplicity': 1,
                'dft_settings': {'functional': 'PBE', 'basis': 'STO-3G'},
                **extra,
            }
        )
    with pytest.raises(ValueError, match='explicit'):
        OrcaCalc(dft_settings={'functional': 'PBE', 'basis': 'STO-3G'})
    with pytest.raises(ValueError):
        OrcaCalc(
            charge=0,
            multiplicity=1,
            dft_settings={'functional': 'PBE SP', 'basis': 'STO-3G'},
        )


@pytest.mark.parametrize(
    'driver,keyword', [('energy', 'SP'), ('electronic', 'SP'), ('opt', 'EnGrad')]
)
def test_actual_writer_execution_and_saved_results(
    structure, tmp_path, fake_engine, capsys, driver, keyword
):
    result_file = tmp_path / 'out.json'
    p = params(structure, result_file, driver, steps=2)
    result = ase_core.run_ase_core(p)
    assert result['status'] == 'success', result
    assert result['converged'] is True
    assert capsys.readouterr().out == ''
    saved = ASEOutputSchema.model_validate_json(result_file.read_text())
    deck = saved.provenance['generated_input']
    assert f'NoAutoStart {keyword}' in deck
    assert 'RKS PBE def2-SVP TightSCF' in deck
    assert 'ConvForced 1' in deck and '%pal nprocs 1 end' in deck
    assert '*xyz 0 1' in deck
    assert (
        'Print[P_OrbEn] 2' in deck
        if driver == 'electronic'
        else 'Print[P_OrbEn]' not in deck
    )
    assert saved.simulation_input.calculator.orcasimpleinput is None
    if driver == 'electronic':
        assert saved.electronic_structure.source == 'orbital_energy_table'
        from chemgraph.tools.report_tools import generate_html

        assert 'Error' not in generate_html.invoke(
            {
                'results_json_path': str(result_file),
                'output_path': str(tmp_path / 'report.html'),
            }
        )


@pytest.mark.parametrize(
    'failure',
    [
        'missing_input',
        'gradient',
        'gradient_energy',
        'timeout',
        'driver',
        'periodic',
        'parity',
        'unknown_element',
        'failed_scf',
        'partial_table',
    ],
)
def test_failures_replace_prior_success(
    structure, tmp_path, monkeypatch, fake_engine, failure
):
    output = tmp_path / 'result.json'
    output.write_text('{"success":true,"old":true}')
    p = params(
        structure,
        output,
        'opt' if failure in {'gradient', 'gradient_energy'} else 'electronic',
    )
    if failure == 'missing_input':
        p.input_structure_file = str(tmp_path / 'missing.xyz')
    elif failure == 'driver':
        p.driver = 'unknown'
    elif failure == 'periodic':
        atoms = read(structure)
        atoms.pbc = True
        write(structure, atoms)
    elif failure == 'parity':
        p.calculator.charge = 1
    elif failure == 'unknown_element':
        write(structure, Atoms('He'))
    else:

        def execute(self, directory, profile):
            fake_engine(self, directory, profile)
            if failure == 'gradient':
                (self.active_directory / 'orca.engrad').unlink()
            elif failure == 'gradient_energy':
                path = self.active_directory / 'orca.engrad'
                text = path.read_text().splitlines()
                text[3] = 'nan'
                path.write_text('\n'.join(text) + '\n')
            elif failure == 'timeout':
                raise TimeoutError('test timeout')
            else:
                path = self.active_directory / 'orca.out'
                text = path.read_text()
                text = (
                    text.replace('SCF CONVERGED AFTER', 'SCF NOT CONVERGED AFTER')
                    if failure == 'failed_scf'
                    else text.replace('29 0.0000', '30 0.0000')
                )
                path.write_text(text)

        monkeypatch.setattr(orca_execution.StructuredOrcaTemplate, 'execute', execute)
    result = ase_core.run_ase_core(p)
    assert result['status'] == 'failure', result
    saved = json.loads(output.read_text())
    assert (
        saved['result_type'] == 'ase_failure'
        and saved['success'] is False
        and 'old' not in saved
    )
    if failure == 'timeout':
        assert saved['stage'] == 'timeout'
        assert saved['artifacts']
    assert ase_core.extract_output_json_core(str(output)) == saved
    from chemgraph.tools.report_tools import generate_html

    assert 'Calculation failed at' in generate_html.invoke(
        {'results_json_path': str(output), 'output_path': str(tmp_path / 'report.html')}
    )
    assert not (tmp_path / 'report.html').exists()


def test_stale_gradient_cannot_satisfy_next_evaluation(
    structure, tmp_path, monkeypatch, fake_engine
):
    def execute(self, directory, profile):
        fake_engine(self, directory, profile)
        gradient = self.active_directory / 'orca.engrad'
        if self.evaluations == 1:
            gradient.write_text(synthetic_engrad(self.atoms, force=0.01))
        else:
            gradient.unlink()

    monkeypatch.setattr(orca_execution.StructuredOrcaTemplate, 'execute', execute)
    result = ase_core.run_ase_core(
        params(structure, tmp_path / 'out.json', 'opt', steps=2)
    )
    assert result['status'] == 'failure'
    assert any('evaluation-0001/orca.engrad' in p for p in result['artifacts'])
    assert any('evaluation-0002/orca.out' in p for p in result['artifacts'])


def test_invalid_driver_before_calculator_and_default_normalization(
    structure, monkeypatch
):
    loader = Mock(side_effect=AssertionError('must not load'))
    monkeypatch.setattr(ase_core, 'load_calculator', loader)
    for model, key in [
        (ASEInputSchema, 'input_structure_file'),
        (ase_input_schema_ensemble, 'input_structure_directory'),
    ]:
        assert (
            model(
                **{key: str(structure), 'calculator': {'calculator_type': 'emt'}}
            ).driver
            == 'opt'
        )
        assert (
            model(
                **{
                    key: str(structure),
                    'calculator': {'calculator_type': 'emt'},
                    'driver': None,
                }
            ).driver
            == 'opt'
        )
        with pytest.raises(ValueError):
            model(
                **{
                    key: str(structure),
                    'calculator': {'calculator_type': 'emt'},
                    'driver': 'bad',
                }
            )
    p = ASEInputSchema(
        input_structure_file=str(structure), calculator={'calculator_type': 'emt'}
    )
    p.driver = 'bad'
    assert ase_core.run_ase_core(p)['status'] == 'failure'
    loader.assert_not_called()


def test_hpc_worker_job_and_calculate_asset_roundtrips(
    structure, tmp_path, fake_engine, monkeypatch
):
    import importlib.util
    from pathlib import Path
    import socket
    from chemgraph.mcp import ase_mcp_hpc

    p = params(structure, tmp_path / 'out.json')
    # The worker uses the same plain job dictionary/model reconstruction as transport.
    assert (
        ASEInputSchema.model_validate(p.model_dump()).calculator.dft_settings.functional
        == 'PBE'
    )
    result = ase_mcp_hpc.run_ase_single(p)
    assert result['status'] == 'success', result
    ensemble = ase_input_schema_ensemble(
        input_structure_directory=str(structure.parent),
        output_results_file=str(tmp_path / 'batch.json'),
        driver='electronic',
        calculator=p.calculator,
    )
    jobs = ase_mcp_hpc._expand_ase_ensemble(ensemble)
    assert jobs
    assert all(
        ASEInputSchema.model_validate(job).calculator.dft_settings.functional == 'PBE'
        for job in jobs
    )
    asset = (
        Path(__file__).parents[1] / 'src/chemgraph/skills/chemgraph/assets/calculate.py'
    )
    spec = importlib.util.spec_from_file_location('calculate_asset', asset)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'input.json').write_text(p.model_dump_json())
    nodes = tmp_path / 'nodes'
    nodes.write_text(socket.gethostname())
    monkeypatch.setenv('PBS_JOBID', 'offline')
    monkeypatch.setenv('PBS_NODEFILE', str(nodes))
    assert module.main() == 0
    p.input_structure_file = str(tmp_path / 'missing.xyz')
    (tmp_path / 'input.json').write_text(p.model_dump_json())
    assert module.main() == 1




def test_publication_error_and_real_process_timeout(structure, tmp_path, monkeypatch):
    from chemgraph.utils.orca import OrcaBudget

    monkeypatch.setattr(orca_execution, 'identify_orca', lambda *a: '6.0.0')
    monkeypatch.setenv('CHEMGRAPH_ORCA_TIMEOUT_S', '1')
    import sys

    binary = tmp_path / 'slow orca'
    binary.write_text(
        f'#!{sys.executable}\nimport time\nprint("diagnostic",flush=True)\ntime.sleep(20)\n'
    )
    binary.chmod(0o755)
    monkeypatch.setenv('ASE_ORCA_COMMAND', str(binary))
    result = ase_core.run_ase_core(params(structure, tmp_path / 'timeout.json'))
    assert result['stage'] == 'timeout'
    assert result['provenance']['evaluation_timeout_s'] == 1
    monkeypatch.setattr(
        ase_core, '_atomic_result', Mock(side_effect=OSError('read-only output'))
    )
    p = params(structure, tmp_path / 'unpublished.json')
    p.driver = 'bad'
    result = ase_core.run_ase_core(p)
    assert 'publication_error' in result and 'results_file' not in result
    assert not (tmp_path / 'unpublished.json').exists()
    # Whole-call deadline expires even if the per-evaluation limit is larger.
    budget = OrcaBudget()
    budget.started -= budget.call_s + 1
    with pytest.raises(TimeoutError):
        budget.remaining()
