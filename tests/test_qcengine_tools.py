import pytest
import numpy as np
from comp_chem_agent.tools.qcengine_tools import (
    is_linear_molecule,
    compute_mass_weighted_hessian,
    build_projection_operator,
    compute_vibrational_frequencies,
    convert_qcmolecule_to_atomsdata,
    convert_atomsdata_to_qcmolecule,
)
from comp_chem_agent.models.atomsdata import AtomsData


@pytest.fixture
def water_molecule():
    """Fixture for water molecule coordinates"""
    return np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.96], [0.0, 0.93, -0.26]]  # O  # H  # H
    )


@pytest.fixture
def co2_molecule():
    """Fixture for CO2 molecule coordinates (linear)"""
    return np.array([[0.0, 0.0, -1.2], [0.0, 0.0, 0.0], [0.0, 0.0, 1.2]])  # O  # C  # O


@pytest.fixture
def sample_hessian():
    """Fixture for a simple 6x6 Hessian matrix (2 atoms)"""
    # Simple harmonic oscillator-like Hessian for H2
    k = 1.0  # force constant
    hessian = np.zeros((6, 6))
    # Set diagonal blocks
    hessian[0:3, 0:3] = np.eye(3) * k
    hessian[3:6, 3:6] = np.eye(3) * k
    # Set off-diagonal blocks
    hessian[0:3, 3:6] = -np.eye(3) * k
    hessian[3:6, 0:3] = -np.eye(3) * k
    return hessian


def test_is_linear_molecule(water_molecule, co2_molecule):
    """Test linear molecule detection"""
    assert not is_linear_molecule(water_molecule)
    assert is_linear_molecule(co2_molecule)

    # Test edge case: single point
    single_point = np.array([[0.0, 0.0, 0.0]])
    assert not is_linear_molecule(single_point)

    # Test diatomic molecule (should be linear)
    diatomic = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert is_linear_molecule(diatomic)


def test_compute_mass_weighted_hessian(sample_hessian):
    """Test mass-weighting of Hessian matrix"""
    masses = [1.0, 1.0]  # masses in amu
    F, mass_vector = compute_mass_weighted_hessian(sample_hessian, masses)

    assert F.shape == sample_hessian.shape
    assert len(mass_vector) == 6  # 2 atoms * 3 coordinates
    assert np.all(mass_vector == mass_vector[0])  # all masses should be equal

    # Test with different masses
    masses = [16.0, 1.0]  # O-H like system
    F, mass_vector = compute_mass_weighted_hessian(sample_hessian, masses)
    assert not np.all(mass_vector[:3] == mass_vector[3:])  # masses should be different


def test_build_projection_operator(water_molecule, co2_molecule):
    """Test building of projection operator"""
    # Test for non-linear molecule (water)
    masses = [16.0, 1.0, 1.0]  # O, H, H
    P_water = build_projection_operator(masses, water_molecule, is_linear=False)
    assert P_water.shape == (9, 9)  # 3 atoms * 3 coordinates

    # Test for linear molecule (CO2)
    masses = [16.0, 12.0, 16.0]  # O, C, O
    P_co2 = build_projection_operator(masses, co2_molecule, is_linear=True)
    assert P_co2.shape == (9, 9)

    # Verify projection properties
    assert np.allclose(P_water @ P_water, P_water)  # idempotent
    assert np.allclose(P_water.T, P_water)  # symmetric


def test_compute_vibrational_frequencies(sample_hessian):
    """Test computation of vibrational frequencies"""
    masses = [1.0, 1.0]  # H2-like system
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    # Test without projection
    freqs = compute_vibrational_frequencies(sample_hessian, masses)
    assert len(freqs) == 3  # 3 vibrational modes

    # Test with projection (linear molecule)
    freqs = compute_vibrational_frequencies(sample_hessian, masses, coords=coords, linear=True)
    assert len(freqs) == 1  # only vibrational mode (others projected out)

    # Test automatic linearity detection
    freqs_auto = compute_vibrational_frequencies(sample_hessian, masses, coords=coords)
    assert np.allclose(freqs, freqs_auto)

    # Test with negative eigenvalue (imaginary frequency)
    negative_hessian = -sample_hessian
    freqs = compute_vibrational_frequencies(negative_hessian, masses)
    assert np.any(freqs < 0)  # should have negative (imaginary) frequencies


def test_edge_cases():
    """Test edge cases and error conditions"""
    # Test with zero Hessian
    zero_hessian = np.zeros((6, 6))
    masses = [1.0, 1.0]
    freqs = compute_vibrational_frequencies(zero_hessian, masses)
    assert len(freqs) == 0  # should return empty array (all frequencies below threshold)

    # Test with single atom
    single_atom_hessian = np.zeros((3, 3))
    masses = [1.0]
    coords = np.array([[0.0, 0.0, 0.0]])
    freqs = compute_vibrational_frequencies(single_atom_hessian, masses, coords=coords)
    assert len(freqs) == 0  # no vibrational modes for single atom


@pytest.fixture
def water_atomsdata():
    """Fixture for water atomsdata"""
    numbers = [8, 1, 1]
    positions = [[0.0, 0.0, 0.0], [0.76, 0.58, 0.0], [-0.76, 0.58, 0.0]]  # Positions in Angstrom
    atomsdata_input = {"numbers": numbers, "positions": positions}
    return AtomsData(**atomsdata_input)


def test_convert_between_atomsdata_and_qcmolecule(water_atomsdata):
    import qcelemental as qcel

    assert isinstance(convert_atomsdata_to_qcmolecule(water_atomsdata), qcel.models.Molecule)
    assert isinstance(
        convert_qcmolecule_to_atomsdata(convert_atomsdata_to_qcmolecule(water_atomsdata)),
        AtomsData,
    )
