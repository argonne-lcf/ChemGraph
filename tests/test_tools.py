import pytest
from chemgraph.tools.ase_tools import (
    run_ase,
    get_symmetry_number,
    is_linear_molecule,
)
from chemgraph.tools.cheminformatics_tools import (
    smiles_to_atomsdata,
    molecule_name_to_smiles,
)
from chemgraph.models.atomsdata import AtomsData
from chemgraph.models.ase_input import ASEOutputSchema, ASEInputSchema


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
    positions = [
        [0.0, 0.0, 0.0],
        [0.76, 0.58, 0.0],
        [-0.76, 0.58, 0.0],
    ]  # Positions in Angstrom
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
    symmetrynumber = get_symmetry_number.invoke({"atomsdata": water_atomsdata})
    assert isinstance(symmetrynumber, int)


def test_is_linear_molecule(water_atomsdata, co2_atomsdata):
    """Test is_linear_molecule function."""
    islinear_water = is_linear_molecule.invoke({"atomsdata": water_atomsdata})
    islinear_co2 = is_linear_molecule.invoke({"atomsdata": co2_atomsdata})
    assert not islinear_water
    assert islinear_co2


@pytest.fixture
def base_ase_input():
    """Base fixture for ASE input with common parameters"""
    return {
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
        "optimizer": "bfgs",
        "calculator": {
            "calculator_type": "emt",
        },
    }


@pytest.fixture
def energy_ase_schema(base_ase_input):
    """Fixture for energy calculation ASE Schema"""
    input_dict = base_ase_input.copy()
    input_dict["driver"] = "energy"
    return ASEInputSchema(**input_dict)


@pytest.fixture
def opt_ase_schema(base_ase_input):
    """Fixture for geometry optimization ASE Schema"""
    input_dict = base_ase_input.copy()
    input_dict["driver"] = "opt"
    return ASEInputSchema(**input_dict)


@pytest.fixture
def vib_ase_schema(base_ase_input):
    """Fixture for vibrational analysis ASE Schema"""
    input_dict = base_ase_input.copy()
    input_dict["driver"] = "vib"
    return ASEInputSchema(**input_dict)


@pytest.fixture
def thermo_ase_schema(base_ase_input):
    """Fixture for thermochemistry ASE Schema"""
    input_dict = base_ase_input.copy()
    input_dict["driver"] = "thermo"
    input_dict["temperature"] = 298
    return ASEInputSchema(**input_dict)


def test_run_ase_energy(energy_ase_schema):
    """Test ASE energy calculation."""
    result = run_ase.invoke({"params": energy_ase_schema})
    assert isinstance(result, ASEOutputSchema)
    assert result.success
    assert result.single_point_energy is not None
    assert result.energy_unit == "eV"


def test_run_ase_opt(opt_ase_schema):
    """Test ASE geometry optimization."""
    result = run_ase.invoke({"params": opt_ase_schema})
    assert isinstance(result, ASEOutputSchema)
    assert result.success
    assert result.converged
    assert result.final_structure is not None
    # Check that optimization changed the structure
    assert result.final_structure.positions != opt_ase_schema.atomsdata.positions


def test_run_ase_vib(vib_ase_schema):
    """Test ASE vibrational analysis."""
    result = run_ase.invoke({"params": vib_ase_schema})
    assert isinstance(result, ASEOutputSchema)
    assert result.success
    assert result.vibrational_frequencies
    assert len(result.vibrational_frequencies) > 0


def test_run_ase_thermo(thermo_ase_schema):
    """Test ASE thermochemistry calculation."""
    result = run_ase.invoke({"params": thermo_ase_schema})
    assert isinstance(result, ASEOutputSchema)
    assert result.success
    assert result.thermochemistry
    # Check for required thermochemistry keys
    assert "enthalpy" in result.thermochemistry
    assert "entropy" in result.thermochemistry
    assert "gibbs_free_energy" in result.thermochemistry
    assert "unit" in result.thermochemistry
    # Check that values are reasonable
    assert result.thermochemistry["unit"] == "eV"
    assert isinstance(result.thermochemistry["enthalpy"], float)
    assert isinstance(result.thermochemistry["entropy"], float)
    assert isinstance(result.thermochemistry["gibbs_free_energy"], float)
    # Check that vibrational frequencies are present
    assert result.vibrational_frequencies
    assert "frequencies" in result.vibrational_frequencies
    assert "frequency_unit" in result.vibrational_frequencies
    assert result.vibrational_frequencies["frequency_unit"] == "cm-1"
