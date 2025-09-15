from pathlib import Path
import json
import pytest
from chemgraph.tools.ase_tools import (
    run_ase,
)
from chemgraph.models.ase_input import ASEInputSchema

TEST_DIR = Path(__file__).parent


@pytest.fixture
def water_xyz_path():
    """Fixture for water.xyz file path located in the same tests/ directory"""
    return Path(__file__).parent / "water.xyz"


@pytest.fixture
def base_ase_input():
    """Base fixture for ASE input with common parameters"""
    return {
        "input_structure_file": str(TEST_DIR / "water.xyz"),
        "output_results_file": str(TEST_DIR / "water_output.json"),
        "optimizer": "bfgs",
        "calculator": {
            "calculator_type": "mace_mp",
        },
    }


@pytest.fixture
def energy_ase_schema(base_ase_input):
    """Fixture for geometry optimization ASE Schema"""
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
    print(result)
    assert isinstance(result, dict)
    assert result['status']
    assert result['single_point_energy'] is not None
    assert result['unit'] == "eV"


def test_run_ase_opt(opt_ase_schema):
    """Test ASE geometry optimization."""
    result = run_ase.invoke({"params": opt_ase_schema})
    assert isinstance(result, dict)
    assert result['status']
    assert result['single_point_energy'] is not None
    assert result['unit'] == "eV"

    # Path to expected output file
    output_file = Path(__file__).parent / "water_output.json"

    # Check file exists
    assert output_file.exists()

    # Optionally validate JSON content
    with open(output_file) as f:
        data = json.load(f)

    assert data["simulation_input"]["driver"] == "opt"


def test_run_ase_vib(vib_ase_schema):
    """Test ASE vibrational analysis."""
    result = run_ase.invoke({"params": vib_ase_schema})
    assert isinstance(result, dict)
    assert result['status']

    # Path to expected output file
    output_file = Path(__file__).parent / "water_output.json"

    # Check file exists
    assert output_file.exists()

    # Optionally validate JSON content
    with open(output_file) as f:
        data = json.load(f)

    assert data["simulation_input"]["driver"] == "vib"
    assert len(data["vibrational_frequencies"]["energies"]) > 0


def test_run_ase_thermo(thermo_ase_schema):
    """Test ASE thermochemistry calculation."""
    result = run_ase.invoke({"params": thermo_ase_schema})

    assert isinstance(result, dict)
    # Path to expected output file
    output_file = Path(__file__).parent / "water_output.json"

    # Check file exists
    assert output_file.exists()

    # Optionally validate JSON content
    with open(output_file) as f:
        data = json.load(f)

    assert data["simulation_input"]["driver"] == "thermo"

    # Check that vibrational frequencies are present
    assert len(data["vibrational_frequencies"]["energies"]) > 0

    # Check for required thermochemistry keys
    assert "enthalpy" in data['thermochemistry']
    assert "entropy" in data['thermochemistry']
    assert "gibbs_free_energy" in data['thermochemistry']
    assert "unit" in data['thermochemistry']
