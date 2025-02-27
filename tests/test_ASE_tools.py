import pytest
import numpy as np
from comp_chem_agent.tools.ASE_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
    geometry_optimization,
)
from comp_chem_agent.models.atomsdata import AtomsData


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


def test_geometry_optimization():
    # Create a simple water molecule for testing
    water_atoms = AtomsData(
        numbers=[8, 1, 1],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        cell=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        pbc=[False, False, False],
    )

    # Test with EMT calculator (faster than MACE for testing)
    result = geometry_optimization.invoke(
        {"atomsdata": water_atoms, "calculator": "emt", "steps": 5}
    )

    assert hasattr(result, "converged")
    assert hasattr(result, "final_structure")
    assert isinstance(result.final_structure, AtomsData)
