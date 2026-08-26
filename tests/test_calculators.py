import importlib.util
import pytest
import numpy as np
from pydantic import ValidationError
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


def test_default_calculator_is_in_detected_available_calculators():
    from chemgraph.schemas.ase_input import (
        get_available_calculator_names,
        get_calculator_selection_context,
        get_default_calculator_name,
    )

    available = get_available_calculator_names()
    default = get_default_calculator_name()
    context = get_calculator_selection_context()

    assert default in available
    assert available
    assert "Calculator availability detected during ChemGraph initialization" in context
    assert default in context


def test_invalid_calculator_type_error_lists_accepted_values():
    # Regression: the error used to list class names (e.g. "EMTCalc"), which
    # misled agents into passing the class name as calculator_type. It should
    # instead name the accepted calculator_type field values (e.g. "emt").
    from chemgraph.schemas.ase_input import ASEInputSchema

    with pytest.raises(ValidationError) as excinfo:
        ASEInputSchema(
            input_structure_file="water.xyz",
            driver="opt",
            calculator={"calculator_type": "EMTCalc"},
        )

    message = str(excinfo.value)
    assert "accepted values" in message
    assert "'emt'" in message


@pytest.mark.skipif(
    importlib.util.find_spec("mace") is None, reason="MACE not installed"
)
@pytest.mark.parametrize("calculator_type", ["mace_mp", "mace_off", "mace_anicc"])
@pytest.mark.parametrize(
    ("schema_name", "structure_input"),
    [
        ("ASEInputSchema", {"input_structure_file": "methane.xyz"}),
        (
            "ase_input_schema_ensemble",
            {"input_structure_directory": "structures"},
        ),
    ],
)
def test_mace_variant_preserved_by_ase_schema(
    calculator_type, schema_name, structure_input
):
    from chemgraph.schemas import ase_input

    schema = getattr(ase_input, schema_name)
    params = schema(
        **structure_input,
        calculator={"calculator_type": calculator_type, "device": "cpu"},
    )

    assert isinstance(params.calculator, MaceCalc)
    assert params.calculator.calculator_type == calculator_type
    assert params.model_dump()["calculator"]["calculator_type"] == calculator_type


@pytest.mark.parametrize("calculator_type", ["mace_invalid", "Mace", "MaceCalc"])
def test_mace_calculator_schema_rejects_invalid_variant(calculator_type):
    with pytest.raises(ValidationError):
        MaceCalc(calculator_type=calculator_type)


@pytest.mark.parametrize(
    ("calculator_type", "model", "expected"),
    [
        ("mace_mp", None, "medium-mpa-0"),
        ("mace_off", None, "medium"),
        ("mace_anicc", None, None),
        ("mace_mp", "small-0b2", "small-0b2"),
    ],
)
def test_mace_model_name_for_output(calculator_type, model, expected):
    calc = MaceCalc(calculator_type=calculator_type, model=model)

    assert calc.get_model_name_for_output() == expected


@pytest.mark.skipif(
    importlib.util.find_spec("mace") is None, reason="MACE not installed"
)
@pytest.mark.parametrize("calculator_type", ["mace_invalid", "Mace", "MaceCalc"])
@pytest.mark.parametrize(
    ("schema_name", "structure_input"),
    [
        ("ASEInputSchema", {"input_structure_file": "methane.xyz"}),
        (
            "ase_input_schema_ensemble",
            {"input_structure_directory": "structures"},
        ),
    ],
)
def test_ase_schema_rejects_invalid_mace_variant(
    calculator_type, schema_name, structure_input
):
    from chemgraph.schemas import ase_input

    schema = getattr(ase_input, schema_name)
    with pytest.raises(ValidationError):
        schema(
            **structure_input,
            calculator={"calculator_type": calculator_type, "device": "cpu"},
        )


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
