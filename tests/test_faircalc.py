import numpy as np
import pytest
from chemgraph.models.calculators.fairchem_calc import FAIRChemCalc
from ase import Atoms
from chemgraph.tools.ase_tools import (
    run_ase,
)
from chemgraph.models.atomsdata import AtomsData
from chemgraph.models.ase_input import ASEOutputSchema, ASEInputSchema


def is_fairchem_installed():
    try:
        import fairchem.core

        return True
    except ImportError:
        return False


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
        "calculator": {"calculator_type": "FAIRChem", "task_name": "omol"},
    }


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


@pytest.mark.skipif(not is_fairchem_installed(), reason="FairChem is not installed")
def test_run_ase_opt(opt_ase_schema):
    """Test ASE geometry optimization."""
    result = run_ase.invoke({"params": opt_ase_schema})
    assert isinstance(result, ASEOutputSchema)
    assert result.success
    assert result.converged
    assert result.final_structure is not None
    # Check that optimization changed the structure
    assert result.final_structure.positions != opt_ase_schema.atomsdata.positions


@pytest.mark.skipif(not is_fairchem_installed(), reason="FairChem is not installed")
def test_run_ase_vib(vib_ase_schema):
    """Test ASE vibrational analysis."""
    result = run_ase.invoke({"params": vib_ase_schema})
    assert isinstance(result, ASEOutputSchema)
    assert result.success
    assert result.vibrational_frequencies
    assert len(result.vibrational_frequencies) > 0


@pytest.mark.skipif(not is_fairchem_installed(), reason="FairChem is not installed")
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
