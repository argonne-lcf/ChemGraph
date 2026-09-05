"""Scientific regressions using EMT and controlled molecular vibrations."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.emt import EMT
from ase.io import write
from ase.thermochemistry import IdealGasThermo
import ase.vibrations

from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.tools.ase_core import get_symmetry_number, run_ase_core


def _run_thermo(tmp_path, monkeypatch, atoms, calculator, temperature, pressure):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    input_path, output_path = tmp_path / "input.xyz", tmp_path / "result.json"
    write(input_path, atoms)
    params = ASEInputSchema(
        input_structure_file=str(input_path), output_results_file=str(output_path),
        calculator=calculator, driver="thermo", temperature=temperature, pressure=pressure,
    )
    result = run_ase_core(params)
    assert result["status"] == "success", result
    output = json.loads(output_path.read_text())
    assert result["result"]["thermochemistry"] == output["thermochemistry"]
    assert output["thermochemistry"]["unit"] == "eV"
    assert output["thermochemistry"]["entropy_unit"] == "eV/K"
    return output


def _assert_reference(actual, reference, temperature, pressure):
    assert actual["enthalpy"] == pytest.approx(reference.get_enthalpy(temperature))
    assert actual["entropy"] == pytest.approx(reference.get_entropy(temperature, pressure))
    assert actual["gibbs_free_energy"] == pytest.approx(reference.get_gibbs_energy(temperature, pressure))


@pytest.mark.parametrize("temperature,pressure", [(250, 101325), (500, 101325), (500, 202650)])
@pytest.mark.parametrize("multiplicity", [1, 2, 3])
def test_monatomic_thermo_matches_ase_without_vibrations(
    tmp_path, monkeypatch, temperature, pressure, multiplicity,
):
    vibrations = Mock(side_effect=AssertionError("an atom has no vibrational modes"))
    monkeypatch.setattr(ase.vibrations, "Vibrations", vibrations)
    monkeypatch.setattr(EMTCalc, "get_multiplicity", lambda self: multiplicity, raising=False)
    atoms = Atoms("Cu", positions=[[0, 0, 0]])
    atoms.calc = EMT()
    potential_energy = atoms.get_potential_energy()
    output = _run_thermo(tmp_path, monkeypatch, atoms, EMTCalc(), temperature, pressure)
    reference = IdealGasThermo(
        vib_energies=[], potentialenergy=potential_energy, atoms=atoms,
        geometry="monatomic", symmetrynumber=1, spin=(multiplicity - 1) / 2,
    )
    _assert_reference(output["thermochemistry"], reference, temperature, pressure)
    assert output["thermochemistry"]["enthalpy"] == pytest.approx(
        potential_energy + 2.5 * units.kB * temperature
    )
    assert output["thermochemistry"]["entropy"] > 0
    assert output["vibrational_frequencies"] == {
        "energies": [], "energy_unit": "meV", "frequencies": [], "frequency_unit": "cm-1",
    }
    vibrations.assert_not_called()
    assert not list(tmp_path.glob("*.traj"))


_MOLECULES = [
    ("OH", [[0, 0, 0], [0, 0, 0.97]], 1, 2, "linear"),
    ("NO", [[0, 0, 0], [0, 0, 1.15]], 1, 2, "linear"),
    ("H2", [[0, 0, 0], [0, 0, 0.74]], 2, 1, "linear"),
    ("OH2", [[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]], 2, 1, "nonlinear"),
]


@pytest.mark.parametrize("symbols,positions,symmetry,multiplicity,geometry", _MOLECULES)
def test_geometric_symmetry_including_radicals(symbols, positions, symmetry, multiplicity, geometry):
    atoms = Atoms(symbols, positions=positions)
    assert get_symmetry_number(AtomsData(numbers=atoms.numbers, positions=positions)) == symmetry


@pytest.mark.parametrize("symbols,positions,symmetry,multiplicity,geometry", _MOLECULES)
def test_molecular_thermo_retains_spin_and_vibrations(
    tmp_path, monkeypatch, symbols, positions, symmetry, multiplicity, geometry,
):
    atoms = Atoms(symbols, positions=positions)
    energies = np.zeros(3 * len(atoms))
    mode_count = 1 if geometry == "linear" else 3
    energies[-mode_count:] = np.arange(1, mode_count + 1) * 0.2

    class ConstantCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
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
    output = _run_thermo(tmp_path, monkeypatch, atoms, config, 300, 101325)
    reference = IdealGasThermo(
        vib_energies=energies, potentialenergy=-1.0, atoms=atoms, geometry=geometry,
        symmetrynumber=symmetry, spin=(multiplicity - 1) / 2,
    )
    _assert_reference(output["thermochemistry"], reference, 300, 101325)
    assert len(vibration_runs) == 1
    assert len(output["vibrational_frequencies"]["energies"]) == mode_count
    assert calculator.atoms.info["charge"] == 0
    assert calculator.atoms.info["spin"] == multiplicity
    assert output["simulation_input"]["calculator"]["multiplicity"] == multiplicity
