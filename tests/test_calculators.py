import importlib.util
import pytest
import numpy as np
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.schemas.calculators.tblite_calc import TBLiteCalc
from chemgraph.schemas.calculators.orca_calc import OrcaCalc
from ase import Atoms


@pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="TBLite not installed"
)
def test_xtb_alias_maps_to_tblite_calculator_schema():
    from chemgraph.schemas.ase_input import ASEInputSchema

    params = ASEInputSchema(
        input_structure_file="methane.xyz",
        driver="ir",
        calculator={"calculator_type": "xTB", "method": "GFN2-xTB"},
    )

    assert params.calculator.calculator_type == "TBLite"
    assert params.calculator.method == "GFN2-xTB"


def test_emt_calculator():
    # Test EMT calculator initialization
    calc = EMTCalc()
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


@pytest.mark.skipif(
    importlib.util.find_spec("mace") is None, reason="MACE not installed"
)
def test_mace_calculator():
    # Test MACE calculator initialization
    calc = MaceCalc(model_type="medium")
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


@pytest.mark.skipif(
    importlib.util.find_spec("tblite") is None, reason="TBLite not installed"
)
def test_tblite_calculator():
    # Test TBLite calculator initialization
    calc = TBLiteCalc()
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


@pytest.mark.skipif(
    importlib.util.find_spec("ase.io.orca") is None, reason="ORCA not installed"
)
def test_orca_calculator():
    # Test ORCA calculator initialization
    from ase.calculators.calculator import BadConfiguration
    from ase import Atoms

    try:
        calc = OrcaCalc()
        ase_calc = calc.get_calculator()
    except BadConfiguration:
        pytest.skip("ORCA calculator not configured in ASE.")

    # Create a simple molecule
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms.calc = ase_calc

    # Test basic calculator properties
    assert hasattr(ase_calc, "calculate")
