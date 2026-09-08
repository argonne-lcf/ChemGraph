"""Scientific regressions using EMT and controlled molecular vibrations."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.io import write
from ase.thermochemistry import IdealGasThermo
import ase.vibrations

from chemgraph.schemas.ase_input import ASEInputSchema, ase_input_schema_ensemble
from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.schemas.calculators.nwchem_calc import NWChemCalc
from chemgraph.tools import ase_core
from chemgraph.tools.ase_core import get_symmetry_number, run_ase_core


def _run_thermo(tmp_path, monkeypatch, atoms, calculator, **conditions):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    input_path, output_path = tmp_path / "input.xyz", tmp_path / "result.json"
    write(input_path, atoms)
    params = ASEInputSchema(
        input_structure_file=str(input_path),
        output_results_file=str(output_path),
        calculator=calculator,
        driver="thermo",
        **conditions,
    )
    result = run_ase_core(params)
    assert result["status"] == "success", result
    output = json.loads(output_path.read_text())
    assert result["result"]["thermochemistry"] == output["thermochemistry"]
    assert output["thermochemistry"]["unit"] == "eV"
    assert output["thermochemistry"]["entropy_unit"] == "eV/K"
    return output


def _assert_reference(actual, reference, temperature, pressure):
    assert actual["enthalpy"] == pytest.approx(
        reference.get_enthalpy(temperature, verbose=False)
    )
    assert actual["entropy"] == pytest.approx(
        reference.get_entropy(temperature, pressure, verbose=False)
    )
    assert actual["gibbs_free_energy"] == pytest.approx(
        reference.get_gibbs_energy(temperature, pressure, verbose=False)
    )


@pytest.mark.parametrize("schema", [ASEInputSchema, ase_input_schema_ensemble])
@pytest.mark.parametrize(
    "conditions,expected",
    [({}, 298.15), ({"temperature": None}, 298.15), ({"temperature": 500}, 500)],
)
def test_temperature_defaults_and_serialization(schema, conditions, expected):
    source = {"input_structure_file": "input.xyz"} if schema is ASEInputSchema else {}
    params = schema(calculator=EMTCalc(), driver="thermo", **source, **conditions)
    assert params.temperature == expected
    assert params.model_dump()["temperature"] == expected
    assert schema.model_json_schema()["properties"]["temperature"]["default"] == 298.15


@pytest.mark.parametrize("schema", [ASEInputSchema, ase_input_schema_ensemble])
@pytest.mark.parametrize(
    "temperature", [0, -1, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_temperature_rejected(schema, temperature):
    source = {"input_structure_file": "input.xyz"} if schema is ASEInputSchema else {}
    with pytest.raises(ValidationError, match="temperature"):
        schema(calculator=EMTCalc(), driver="thermo", temperature=temperature, **source)


@pytest.mark.parametrize("conditions", [{}, {"temperature": None}])
@pytest.mark.parametrize("calc_model", [EMTCalc(), NWChemCalc()])
def test_atomic_default_temperature_and_singlet_warning(
    tmp_path,
    monkeypatch,
    capsys,
    caplog,
    conditions,
    calc_model,
):
    atoms = Atoms("Cu", positions=[[0, 0, 0]])
    atoms.calc = EMT()
    # Exercise both an absent accessor and an accessor returning None without
    # starting an external engine or adding methods to a real calculator schema.
    monkeypatch.setattr(
        ase_core, "load_calculator", lambda config: (EMT(), {}, calc_model)
    )
    output = _run_thermo(tmp_path, monkeypatch, atoms, EMTCalc(), **conditions)
    assert output["simulation_input"]["temperature"] == 298.15
    assert capsys.readouterr().out == ""
    assert "assuming a singlet" in caplog.text
    reference = IdealGasThermo(
        vib_energies=[],
        potentialenergy=atoms.get_potential_energy(),
        atoms=atoms,
        geometry="monatomic",
        symmetrynumber=1,
        spin=0,
    )
    _assert_reference(output["thermochemistry"], reference, 298.15, 101325)


@pytest.mark.parametrize(
    "temperature,pressure", [(250, 101325), (500, 101325), (500, 202650)]
)
@pytest.mark.parametrize("multiplicity", [1, 2, 3])
def test_monatomic_thermo_matches_ase_without_vibrations(
    tmp_path,
    monkeypatch,
    capsys,
    caplog,
    temperature,
    pressure,
    multiplicity,
):
    vibrations = Mock(side_effect=AssertionError("an atom has no vibrational modes"))
    monkeypatch.setattr(ase.vibrations, "Vibrations", vibrations)
    monkeypatch.setattr(MaceCalc, "get_calculator", lambda self: EMT())
    atoms = Atoms("Cu", positions=[[0, 0, 0]])
    atoms.calc = EMT()
    potential_energy = atoms.get_potential_energy()
    config = MaceCalc(calculator_type="mace_polar", multiplicity=multiplicity)
    output = _run_thermo(
        tmp_path,
        monkeypatch,
        atoms,
        config,
        temperature=temperature,
        pressure=pressure,
    )
    assert capsys.readouterr().out == ""
    assert "assuming a singlet" not in caplog.text
    reference = IdealGasThermo(
        vib_energies=[],
        potentialenergy=potential_energy,
        atoms=atoms,
        geometry="monatomic",
        symmetrynumber=1,
        spin=(multiplicity - 1) / 2,
    )
    _assert_reference(output["thermochemistry"], reference, temperature, pressure)
    assert output["thermochemistry"]["enthalpy"] == pytest.approx(
        potential_energy + 2.5 * units.kB * temperature
    )
    assert output["thermochemistry"]["entropy"] > 0
    assert output["vibrational_frequencies"] == {
        "energies": [],
        "energy_unit": "meV",
        "frequencies": [],
        "frequency_unit": "cm-1",
    }
    vibrations.assert_not_called()
    assert not list(tmp_path.glob("*.traj"))
    assert not list(tmp_path.glob("frequencies_*.csv"))


@pytest.mark.parametrize("driver", ["thermo", "vib", "ir"])
@pytest.mark.parametrize("stem", ["input", "input[1]"])
def test_atomic_drivers_skip_displacements_and_clean_artifacts(
    tmp_path,
    monkeypatch,
    driver,
    stem,
):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    vibrations, infrared = Mock(), Mock()
    monkeypatch.setattr(ase.vibrations, "Vibrations", vibrations)
    monkeypatch.setattr(ase.vibrations, "Infrared", infrared)
    stale = [f"frequencies_{stem}.csv", f"{stem}_vib.5.traj"]
    ir_files = [
        f"ir_spectrum_{stem}.png",
        f"ir_spectrum_{stem}.csv",
        f"ir_peaks_{stem}.csv",
    ]
    unrelated = [
        "frequencies_other.csv",
        "other_vib.5.traj",
        "ir_spectrum_other.csv",
        "input1_vib.5.traj",
    ]
    for name in stale + ir_files + unrelated:
        (tmp_path / name).write_text("previous calculation")
    write(tmp_path / f"{stem}.xyz", Atoms("Cu", positions=[[0, 0, 0]]))
    result = run_ase_core(
        ASEInputSchema(
            input_structure_file=str(tmp_path / f"{stem}.xyz"),
            output_results_file="result.json",
            calculator=EMTCalc(),
            driver=driver,
        )
    )
    assert result["status"] == "success", result
    output = json.loads((tmp_path / "result.json").read_text())
    assert output["vibrational_frequencies"] == {
        "energies": [],
        "energy_unit": "meV",
        "frequencies": [],
        "frequency_unit": "cm-1",
    }
    vibrations.assert_not_called()
    infrared.assert_not_called()
    assert all(not (tmp_path / name).exists() for name in stale)
    assert all(
        (tmp_path / name).read_text() == "previous calculation" for name in unrelated
    )
    if driver == "ir":
        assert (
            "Single atoms have no vibrational modes or IR spectrum" in result["message"]
        )
        assert "saved as individual .traj files" not in result["message"]
        assert all(not (tmp_path / name).exists() for name in ir_files)
        assert output["ir_data"] == {
            "spectrum_frequencies": [],
            "spectrum_frequencies_units": "cm-1",
            "spectrum_intensities": [],
            "spectrum_intensities_units": "D/Å^2 amu^-1",
        }
    else:
        assert all((tmp_path / name).exists() for name in ir_files)


@pytest.mark.parametrize("number", [1, 8, 29])
def test_atomic_symmetry_is_one(number):
    assert get_symmetry_number(AtomsData(numbers=[number], positions=[[0, 0, 0]])) == 1


_MOLECULES = [
    ("OH", [[0, 0, 0], [0, 0, 0.97]], 1, 2, "linear"),
    ("NO", [[0, 0, 0], [0, 0, 1.15]], 1, 2, "linear"),
    ("H2", [[0, 0, 0], [0, 0, 0.74]], 2, 1, "linear"),
    ("OH2", [[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]], 2, 1, "nonlinear"),
    ("CO2", [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]], 2, 1, "linear"),
]


@pytest.mark.parametrize("symbols,positions,symmetry,multiplicity,geometry", _MOLECULES)
def test_geometric_symmetry_including_radicals(
    symbols, positions, symmetry, multiplicity, geometry
):
    atoms = Atoms(symbols, positions=positions)
    assert (
        get_symmetry_number(AtomsData(numbers=atoms.numbers, positions=positions))
        == symmetry
    )


@pytest.mark.parametrize("symbols,positions,symmetry,multiplicity,geometry", _MOLECULES)
def test_molecular_thermo_retains_spin_and_vibrations(
    tmp_path,
    monkeypatch,
    capsys,
    symbols,
    positions,
    symmetry,
    multiplicity,
    geometry,
):
    atoms = Atoms(symbols, positions=positions)
    energies = np.full(3 * len(atoms), 1e-7j, dtype=complex)
    mode_count = 3 * len(atoms) - (5 if geometry == "linear" else 6)
    energies[-mode_count:] = np.arange(1, mode_count + 1) * 0.2

    class ConstantCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(
            self, atoms=None, properties=("energy",), system_changes=all_changes
        ):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": -1.0, "forces": np.zeros((len(atoms), 3))}

    calculator = ConstantCalculator()
    monkeypatch.setattr(MaceCalc, "get_calculator", lambda self: calculator)
    vibration_runs = []

    class ControlledVibrations:
        def __init__(self, atoms, name):
            self.atoms, self.name = atoms, name

        def clean(self):
            pass

        def run(self):
            vibration_runs.append(self.atoms)

        def get_energies(self):
            return energies

        def write_mode(self, n, **kwargs):
            write(Path(f"{self.name}.{n}.traj"), self.atoms)

    monkeypatch.setattr(ase.vibrations, "Vibrations", ControlledVibrations)
    config = MaceCalc(calculator_type="mace_polar", multiplicity=multiplicity, charge=0)
    output = _run_thermo(
        tmp_path, monkeypatch, atoms, config, temperature=300, pressure=101325
    )
    stdout = capsys.readouterr().out
    assert "Enthalpy components" not in stdout
    assert "Entropy components" not in stdout
    assert "Free energy components" not in stdout
    reference = IdealGasThermo(
        vib_energies=energies[-mode_count:].real,
        potentialenergy=-1.0,
        atoms=atoms,
        geometry=geometry,
        symmetrynumber=symmetry,
        spin=(multiplicity - 1) / 2,
    )
    _assert_reference(output["thermochemistry"], reference, 300, 101325)
    assert len(vibration_runs) == 1
    assert len(output["vibrational_frequencies"]["energies"]) == mode_count
    assert calculator.atoms.info["charge"] == 0
    assert calculator.atoms.info["spin"] == multiplicity
    assert output["simulation_input"]["calculator"]["multiplicity"] == multiplicity
