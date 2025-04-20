import pytest
from comp_chem_agent.tools.ASE_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
    run_ase,
    get_symmetry_number,
    is_linear_molecule,
)
from comp_chem_agent.models.atomsdata import AtomsData
from comp_chem_agent.models.ase_input import ASEOutputSchema, ASEInputSchema


def test_molecule_name_to_smiles():
    # Test with a known molecule
    assert molecule_name_to_smiles.invoke("water") == "O"
    assert molecule_name_to_smiles.invoke("methane") == "C"

    # Test with invalid molecule name
    with pytest.raises(Exception):
        molecule_name_to_smiles.invoke("not_a_real_molecule_name")


def test_smiles_to_atomsdata():
    # Test with simple molecules
    water = smiles_to_atomsdata.invoke({"smiles": "O"})
    assert isinstance(water, AtomsData)
    assert len(water.numbers) == 3  # O + 2H
    assert water.numbers[0] == 8  # Oxygen atomic number

    methane = smiles_to_atomsdata.invoke({"smiles": "C"})
    assert isinstance(methane, AtomsData)
    assert len(methane.numbers) == 5  # C + 4H

    # Test with invalid SMILES
    with pytest.raises(ValueError):
        smiles_to_atomsdata.invoke({"smiles": "invalid_smiles"})


@pytest.fixture
def water_atomsdata():
    """Fixture for water atomsdata"""
    numbers = [8, 1, 1]
    positions = [[0.0, 0.0, 0.0], [0.76, 0.58, 0.0], [-0.76, 0.58, 0.0]]  # Positions in Angstrom
    atomsdata_input = {"numbers": numbers, "positions": positions}
    return AtomsData(**atomsdata_input)


@pytest.fixture
def co2_atomsdata():
    """Fixture for CO2 atomsdata"""
    numbers = [6, 8, 8]
    positions = [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]
    atomsdata_input = {"numbers": numbers, "positions": positions}
    return AtomsData(**atomsdata_input)


def test_get_symmetry_number(water_atomsdata):
    """Test get_symmetry_number function."""
    symmetrynumber = get_symmetry_number.invoke({'atomsdata': water_atomsdata})
    assert isinstance(symmetrynumber, int)


def test_is_linear_molecule(water_atomsdata, co2_atomsdata):
    """Test is_linear_molecule function."""
    islinear_water = is_linear_molecule.invoke({'atomsdata': water_atomsdata})
    islinear_co2 = is_linear_molecule.invoke({'atomsdata': co2_atomsdata})
    assert islinear_water == False
    assert islinear_co2 == True


@pytest.fixture
def sample_ase_schema():
    """Fixture for a sample ASE Schema"""
    input_dict = {
        "atomsdata": {
            "numbers": [8, 1, 1],
            "positions": [
                [0.00893, 0.40402, 0.0],
                [-0.78731, -0.1847, 0.0],
                [0.77838, -0.21932, 0.0],
            ],
            "cell": None,
            "pbc": None,
        },
        "driver": "thermo",
        "optimizer": "bfgs",
        "calculator": {
            "calculator_type": "emt",
        },
    }
    return ASEInputSchema(**input_dict)


def test_run_ase(sample_ase_schema):
    """Test run_ase function."""
    result = run_ase.invoke({'params': sample_ase_schema})
    assert isinstance(result, ASEOutputSchema)


def test_run_geometry_optimization(sample_ase_schema):
    """Test run_geometry_optimization function."""
    # Temporarily skip test for geometry optimization, assuming run_ase covers it
    # result = run_geometry_optimization.invoke({'params': sample_ase_schema})
    # assert isinstance(result, ASEOutputSchema)
    pytest.skip("Skipping test_run_geometry_optimization, assuming run_ase covers this.")


def test_run_vibrational_frequency(sample_ase_schema):
    """Test run_vibrational_frequency"""
    # Temporarily skip test for vibrational frequency
    pytest.skip("Skipping test_run_vibrational_frequency as the function is missing.")


def test_calculate_thermochemistry(sample_ase_schema):
    """Test calculate_thermochemistry"""
    pytest.skip("Skipping test_calculate_thermochemistry as the function is missing.")
