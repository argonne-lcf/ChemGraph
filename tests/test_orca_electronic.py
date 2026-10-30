"""Opt-in live engine proof. No ORCA is installed or downloaded by these tests."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('CHEMGRAPH_TEST_ORCA') != '1',
    reason='Set CHEMGRAPH_TEST_ORCA=1 for licensed local ORCA',
)


def test_live_orca_electronic(tmp_path):
    from ase.io import read, write
    from ase.units import Hartree
    from chemgraph.utils.orca import resolve_orca
    from chemgraph.schemas.ase_input import ASEInputSchema
    from chemgraph.tools.ase_core import run_ase_core
    from chemgraph.tools.molecular_core import sha256_file

    resolve_orca()  # An opted-in missing/misconfigured engine must fail, not skip.
    atoms = read(
        Path(__file__).parent / 'data/orca/dvb_sp_dft.out', format='orca-output'
    )
    structure = tmp_path / 'dvb.xyz'
    write(structure, atoms)
    result = run_ase_core(
        ASEInputSchema(
            input_structure_file=str(structure),
            output_results_file=str(tmp_path / 'electronic.json'),
            driver='electronic',
            calculator={
                'calculator_type': 'orca',
                'charge': 0,
                'multiplicity': 1,
                'dft_settings': {'functional': 'B3LYP', 'basis': 'STO-3G'},
            },
        )
    )
    assert result['status'] == 'success', result
    record = result['electronic_structure']
    assert record['source'] == 'orbital_energy_table'
    assert record['nbf'] == record['nmo'] == 60 and record['electron_count'] == 70
    assert record['total_energy'] / Hartree == pytest.approx(
        -382.055108392562, abs=1e-3
    )
    assert record['homo'] == pytest.approx(-0.14930 * Hartree, abs=0.02)
    assert record['lumo'] == pytest.approx(0.04125 * Hartree, abs=0.02)
    # Preserve native full-table proof and provenance separately from upstream fixtures.
    (tmp_path / 'live-provenance.json').write_text(
        json.dumps(
            {
                'kind': 'live ORCA, generated TightSCF deck',
                'engine_version': record['engine_version'],
                'artifacts': record['artifacts'],
                'result_sha256': sha256_file(tmp_path / 'electronic.json'),
            },
            indent=2,
        )
    )


def test_live_orca_optimization_and_full_spectrum(tmp_path):
    from chemgraph.utils.orca import resolve_orca
    from examples.molecular_properties.run import run_workflow

    resolve_orca()
    summary, code = run_workflow(
        'CN',
        'PBE',
        'def2-TZVP',
        str(tmp_path),
        preparation_settings={'candidates': 2, 'retained': 1},
    )
    assert code == 0, summary
    row = summary['conformers'][0]
    data = json.loads(Path(row['artifacts']['electronic']).read_text())
    record = data['electronic_structure']
    assert record['scf_converged'] and record['source'] == 'orbital_energy_table'
    assert sum(o['occupation'] == 0 for o in record['orbitals']) > 20
    assert [o['index'] for o in record['orbitals']] == list(range(record['nbf']))
    assert (
        data['simulation_input']['input_structure_file']
        == row['artifacts']['optimized_xyz']
    )
