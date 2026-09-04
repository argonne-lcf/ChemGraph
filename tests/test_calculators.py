import importlib.util
import subprocess
import sys

import pytest
import numpy as np
from pydantic import ValidationError
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.schemas.calculators.tblite_calc import TBLiteCalc
from chemgraph.schemas.calculators.orca_calc import OrcaCalc
from ase import Atoms


def test_ase_schema_import_does_not_load_optional_native_engines():
    """Schema discovery must not initialize mutually incompatible runtimes."""
    script = """
import sys
import chemgraph.schemas.ase_input  # noqa: F401

optional_modules = ("torch", "tblite.ase", "fairchem.core", "mace")
loaded = [name for name in optional_modules if name in sys.modules]
if loaded:
    raise SystemExit(f"optional native modules loaded: {loaded}")
"""

    subprocess.run([sys.executable, "-c", script], check=True)


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
    if importlib.util.find_spec("mace") is not None:
        assert default == "MaceCalc"
        assert "calculator_type='mace_polar'" in context
        assert "model='polar-1-m'" in context

        from chemgraph.schemas.ase_input import ASEInputSchema

        params = ASEInputSchema(input_structure_file="water.xyz", driver="energy")
        assert isinstance(params.calculator, MaceCalc)
        assert params.calculator.calculator_type == "mace_polar"
        assert params.calculator.get_model_name_for_output() == "polar-1-m"


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
@pytest.mark.parametrize(
    "calculator_type", ["mace_polar", "mace_mp", "mace_off", "mace_anicc"]
)
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
        ("mace_polar", None, "polar-1-m"),
        ("mace_mp", None, "medium-mpa-0"),
        ("mace_off", None, "medium"),
        ("mace_anicc", None, None),
        ("mace_mp", "small-0b2", "small-0b2"),
    ],
)
def test_mace_model_name_for_output(calculator_type, model, expected):
    calc = MaceCalc(calculator_type=calculator_type, model=model)

    assert calc.get_model_name_for_output() == expected


def test_mace_polar_medium_is_default():
    calc = MaceCalc()

    assert calc.calculator_type == "mace_polar"
    assert calc.get_model_name_for_output() == "polar-1-m"
    assert calc.get_atoms_properties() == {
        "charge": 0,
        "spin": 1,
        "external_field": [0.0, 0.0, 0.0],
    }
    description = MaceCalc.model_fields["calculator_type"].description
    assert description is not None
    assert "dipole moments" in description


def test_nonpolar_mace_does_not_inject_polar_metadata():
    assert MaceCalc(calculator_type="mace_mp").get_atoms_properties() == {}


@pytest.mark.skipif(
    importlib.util.find_spec("mace") is None, reason="MACE not installed"
)
@pytest.mark.parametrize(
    ("model", "expected_model"), [(None, "polar-1-m"), ("polar-1-l", "polar-1-l")]
)
def test_mace_polar_loader_uses_selected_model(monkeypatch, model, expected_model):
    import mace.calculators

    class FakePolarCalculator:
        implemented_properties = ["energy"]

    sentinel = FakePolarCalculator()
    received = []

    def fake_mace_polar(**kwargs):
        received.append(kwargs)
        return sentinel

    monkeypatch.setattr(mace.calculators, "mace_polar", fake_mace_polar)

    calc = MaceCalc(model=model, device="cpu", default_dtype="float64")

    assert calc.get_calculator() is sentinel
    assert received == [
        {"model": expected_model, "device": "cpu", "default_dtype": "float64"}
    ]
    assert "dipole" in sentinel.implemented_properties


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
def test_mace_calculator(monkeypatch):
    # Exercise the ASE calculator interface without downloading model weights.
    import mace.calculators
    from ase.calculators.emt import EMT

    monkeypatch.setattr(mace.calculators, "mace_polar", lambda **kwargs: EMT())

    calc = MaceCalc()
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
