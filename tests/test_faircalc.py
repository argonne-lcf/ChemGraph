import numpy as np
from chemgraph.models.calculators.fair_calc import FAIRChemCalc
from ase import Atoms


def test_fair_calculator():
    # Test EMT calculator initialization
    calc = FAIRChemCalc()
    ase_calc = calc.get_calculator()

    # Create a simple molecule
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.calc = ase_calc

    # Test energy calculation
    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)

    # Test forces calculation
    forces = atoms.get_forces()
    assert isinstance(forces, np.ndarray)
    assert forces.shape == (2, 3)
