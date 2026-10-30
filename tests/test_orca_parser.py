"""Genuine fixture coverage and explicitly synthetic corruptions of native evidence."""

import hashlib
import json
from pathlib import Path
import re

import pytest
from ase.io import read
from ase.units import Hartree

from chemgraph.parsers.orca import OrcaParseError, parse_orca_output

DATA = Path(__file__).parent / 'data' / 'orca'


@pytest.fixture
def native():
    output = DATA / 'dvb_sp_dft.out'
    atoms = read(output, format='orca-output')
    return output.read_text(), dict(
        charge=0,
        multiplicity=1,
        symbols=atoms.get_chemical_symbols(),
        positions=atoms.positions,
        method='B3LYP',
        basis='STO-3G',
        input_path=output.with_suffix('.inp'),
        output_path=output,
    )


def test_genuine_fixture_bytes_and_complete_mo_headers(native):
    provenance = json.loads((DATA / 'provenance.json').read_text())
    for name, evidence in provenance['files'].items():
        assert (
            hashlib.sha256((DATA / name).read_bytes()).hexdigest() == evidence['sha256']
        )
    text, kwargs = native
    record = parse_orca_output(text, **kwargs)
    assert record.source == 'mo_header'
    assert record.nbf == record.nmo == 60
    assert record.electron_count == sum(o.occupation for o in record.orbitals) == 70
    assert record.homo_indices == [34] and record.lumo_indices == [35]
    assert record.homo == pytest.approx(-0.14930 * Hartree)
    assert record.lumo == pytest.approx(0.04125 * Hartree)
    assert record.total_energy == pytest.approx(-382.055108392562 * Hartree)
    assert record.native_settings['COSX GRID GENERATION'] != 'not_reported'


def test_genuine_61_rejects_native_nmr_before_spectrum(native):
    _, kwargs = native
    output = DATA / 'dvb_coupling_nmr.out'
    with pytest.raises(OrcaParseError, match='unsupported_native_task'):
        parse_orca_output(output.read_text(), **{**kwargs, 'output_path': output})


@pytest.mark.parametrize(
    'mutation,reason',
    [
        (
            lambda s: (
                s[: s.index('MOLECULAR ORBITALS')]
                + s[s.index('MULLIKEN ATOMIC CHARGES') :]
            ),
            'incomplete_spectrum',
        ),
        (
            lambda s: s.replace('SCF CONVERGED AFTER', 'SCF OMITTED AFTER'),
            'missing_or_ambiguous_evidence',
        ),
        (lambda s: s.replace('ORCA TERMINATED NORMALLY', ''), 'abnormal_termination'),
        (
            lambda s: s.replace('Program Version 6.0.0', 'Program Version 5.0.0'),
            'unsupported_version',
        ),
        (
            lambda s: s.replace('Program Version 6.0.0', ''),
            'missing_or_ambiguous_evidence',
        ),
        (lambda s: s + '\nSCF NOT CONVERGED\n', 'scf_failure'),
        (
            lambda s: s.replace(
                'Number of eigenvalues below threshold      ... 0',
                'Number of eigenvalues below threshold      ... 1',
            ),
            'unsupported_basis_reduction',
        ),
        (
            lambda s: s.replace(
                'Number of eigenvalues below threshold', 'Unknown overlap diagnostic'
            ),
            'unsupported_basis_reduction',
        ),
        (
            lambda s: s.replace(
                'Charge          ....    0', 'Charge          ....    1'
            ),
            'identity_mismatch',
        ),
        (
            lambda s: s.replace(
                'NEL             ....   70', 'NEL             ....   68'
            ),
            'identity_mismatch',
        ),
        (
            lambda s: s.replace('HFTyp           .... RHF', 'HFTyp           .... UHF'),
            'unsupported_spin',
        ),
        (
            lambda s: s.replace('-10.01641 -10.01636', '-11.01641 -10.01636'),
            'inconsistent_spectrum',
        ),
        (
            lambda s: s.replace(
                '                      0         1',
                '                      0         0',
                1,
            ),
            'invalid_spectrum',
        ),
        (
            lambda s: s.replace(' -1.415253    0.230222', ' -2.415253    0.230222'),
            'geometry_mismatch',
        ),
        (
            lambda s: s + '\nFINAL SINGLE POINT ENERGY -382.0\n',
            'unsupported_native_task',
        ),
        (
            lambda s: s.replace(
                '! rks b3lyp sto-3g usesym', '! rks b3lyp sto-3g usesym opt'
            ),
            'unsupported_native_task',
        ),
    ],
)
def test_labeled_synthetic_corruptions(native, tmp_path, mutation, reason):
    text, kwargs = native
    modified = mutation(text)
    assert modified != text
    output = tmp_path / 'synthetic.out'
    output.write_text(modified)
    with pytest.raises(ValueError, match=reason):
        parse_orca_output(modified, **{**kwargs, 'output_path': output})


def test_full_synthetic_table_preferred_and_consistency(native, tmp_path):
    from chemgraph.parsers.orca import _mo_headers

    text, kwargs = native
    rows = _mo_headers(text)
    full = '\n'.join(f'{i} {o:.4f} {e:.5f} {e * 27.2113834:.4f}' for i, o, e, _ in rows)
    modified = re.sub(
        r'(ORBITAL ENERGIES\n-+\n).*?(\n-+\nMOLECULAR ORBITALS)',
        r'\g<1>\n' + full + r'\g<2>',
        text,
        flags=re.S,
    )
    output = tmp_path / 'synthetic-full.out'
    output.write_text(modified)
    record = parse_orca_output(modified, **{**kwargs, 'output_path': output})
    assert record.source == 'orbital_energy_table' and len(record.orbitals) == 60
    for corrupt in [
        modified.replace('59 0.0000', '59 1.0000'),
        modified.replace('59 0.0000', '58 0.0000'),
    ]:
        output.write_text(corrupt)
        with pytest.raises(ValueError):
            parse_orca_output(corrupt, **{**kwargs, 'output_path': output})








@pytest.mark.parametrize(
    'directive', ['CPCM(water)', 'NMR', 'DLPNO-CCSD(T)', 'EPR', 'NumGrad']
)
def test_synthetic_unsupported_task_keywords_do_not_become_gas_phase_records(
    native, directive
):
    text, kwargs = native
    text = text.replace(
        '! rks b3lyp sto-3g usesym', '! rks b3lyp sto-3g usesym ' + directive
    )
    with pytest.raises(ValueError, match='unsupported_native_task'):
        parse_orca_output(text, **kwargs)
