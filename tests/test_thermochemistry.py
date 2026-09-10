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
from ase.io import read, write
from ase.thermochemistry import IdealGasThermo
import ase.vibrations

from chemgraph.schemas.ase_input import ASEInputSchema, ase_input_schema_ensemble
from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.calculators.emt_calc import EMTCalc
from chemgraph.schemas.calculators.mace_calc import MaceCalc
from chemgraph.schemas.calculators.nwchem_calc import NWChemCalc
from chemgraph.tools import ase_core
from chemgraph.tools.ase_core import get_symmetry_number, run_ase_core


def _controlled_spectrum(monkeypatch, energies):
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
            return np.asarray(energies, dtype=complex)

        def write_mode(self, n, **kwargs):
            frame = self.atoms.copy()
            frame.info["mode_index"] = n
            write(Path(f"{self.name}.{n}.traj"), frame)

    monkeypatch.setattr(ase.vibrations, "Vibrations", ControlledVibrations)
    return calculator, vibration_runs


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
    output = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["result"]["thermochemistry"] == output["thermochemistry"]
    assert output["thermochemistry"]["unit"] == "eV"
    assert output["thermochemistry"]["entropy_unit"] == "eV/K"
    assert output["thermochemistry"]["ignore_imag_modes"] is False
    assert output["thermochemistry"]["n_imag"] == 0
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
        "mode_indices": [],
        "all_modes": [],
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
    output = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    expected_vibrations = {
        "energies": [],
        "energy_unit": "meV",
        "frequencies": [],
        "frequency_unit": "cm-1",
    }
    if driver == "thermo":
        expected_vibrations.update(mode_indices=[], all_modes=[])
    assert output["vibrational_frequencies"] == expected_vibrations
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


@pytest.fixture
def water_vibration_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    write(
        tmp_path / "water.xyz",
        Atoms("OH2", positions=[[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]]),
    )
    params = ASEInputSchema(
        input_structure_file="water.xyz",
        output_results_file="result.json",
        calculator=EMTCalc(),
        driver="vib",
    )
    result = run_ase_core(params)
    assert result["status"] == "success", result
    paths = [tmp_path / "frequencies_water.csv"] + [
        tmp_path / f"water_vib.{i}.traj" for i in (6, 7, 8)
    ]
    # EMT has no dipole implementation; seed prior IR outputs separately.
    for name in (
        "ir_spectrum_water.png", "ir_spectrum_water.csv", "ir_peaks_water.csv",
    ):
        path = tmp_path / name
        path.write_bytes(b"previous IR calculation")
        paths.append(path)
    return params, {path: path.read_bytes() for path in paths}


@pytest.mark.parametrize("driver", ["vib", "thermo", "ir"])
def test_displacement_failure_preserves_previous_artifacts(
    monkeypatch, water_vibration_artifacts, driver,
):
    params, artifacts = water_vibration_artifacts
    original_run = ase.vibrations.Vibrations.run
    evaluations = 0

    def fail_during_displacements(vib):
        original_calculate = vib.atoms.calc.calculate

        def calculate(*args, **kwargs):
            nonlocal evaluations
            evaluations += 1
            if evaluations > 10:
                raise RuntimeError("SCF did not converge")
            return original_calculate(*args, **kwargs)

        monkeypatch.setattr(vib.atoms.calc, "calculate", calculate)
        return original_run(vib)

    monkeypatch.setattr(ase.vibrations.Vibrations, "run", fail_during_displacements)
    result = run_ase_core(params.model_copy(update={"driver": driver}))
    assert result["status"] == "failure", result
    assert result["message"] == "SCF did not converge"
    assert evaluations == 11
    assert all(path.exists() for path in artifacts)
    assert {path: path.read_bytes() for path in artifacts} == artifacts


def test_mode_write_failure_preserves_previous_trajectories(
    monkeypatch, water_vibration_artifacts,
):
    params, artifacts = water_vibration_artifacts
    trajectories = {path: data for path, data in artifacts.items() if path.suffix == ".traj"}
    original_write_mode = ase.vibrations.Vibrations.write_mode
    written_modes = []

    def fail_after_first_mode(vib, n, **kwargs):
        if written_modes:
            raise RuntimeError("mode write failed")
        original_write_mode(vib, n=n, **kwargs)
        written_modes.append(n)

    monkeypatch.setattr(ase.vibrations.Vibrations, "write_mode", fail_after_first_mode)
    result = run_ase_core(params)
    assert result["status"] == "failure", result
    assert result["message"] == "mode write failed"
    assert written_modes == [6]
    assert all(path.exists() for path in trajectories)
    assert {path: path.read_bytes() for path in trajectories} == trajectories


def test_ir_failure_preserves_previous_ir_artifacts(
    monkeypatch, water_vibration_artifacts,
):
    params, artifacts = water_vibration_artifacts
    ir_artifacts = {path: data for path, data in artifacts.items() if path.name.startswith("ir_")}
    infrared_run = Mock(side_effect=RuntimeError("IR calculation failed"))
    monkeypatch.setattr(ase.vibrations.Infrared, "run", infrared_run)
    result = run_ase_core(params.model_copy(update={"driver": "ir"}))
    assert result["status"] == "failure", result
    assert result["message"] == "IR calculation failed"
    infrared_run.assert_called_once()
    assert all(path.exists() for path in ir_artifacts)
    assert {path: path.read_bytes() for path in ir_artifacts} == ir_artifacts


@pytest.mark.parametrize("driver", ["vib", "thermo", "ir"])
@pytest.mark.parametrize("stem", ["water", "water[1]"])
def test_molecular_rerun_replaces_artifacts_and_removes_obsolete_modes(
    tmp_path, monkeypatch, driver, stem,
):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    energies = [0.001] * 6 + [0.02, 0.03, 0.04]
    _controlled_spectrum(monkeypatch, energies)
    if driver == "ir":
        infrared = Mock()
        infrared.get_spectrum.return_value = ([500, 1000], [0.2, 0.1])
        infrared.get_energies.return_value = np.asarray(energies, dtype=complex)
        infrared.intensities = np.arange(9, dtype=float)
        monkeypatch.setattr(ase.vibrations, "Infrared", Mock(return_value=infrared))
    write(
        tmp_path / f"{stem}.xyz",
        Atoms("OH2", positions=[[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]]),
    )
    replacements = [f"frequencies_{stem}.csv"] + [
        f"{stem}_vib.{i}.traj" for i in (6, 7, 8)
    ]
    if driver == "ir":
        replacements += [
            f"ir_spectrum_{stem}.png", f"ir_spectrum_{stem}.csv", f"ir_peaks_{stem}.csv",
        ]
    obsolete = f"{stem}_vib.5.traj"
    unrelated = ["other_vib.5.traj", "water1_vib.5.traj", "frequencies_other.csv"]
    for name in replacements + [obsolete] + unrelated:
        (tmp_path / name).write_bytes(b"previous calculation")
    result = run_ase_core(
        ASEInputSchema(
            input_structure_file=f"{stem}.xyz",
            output_results_file="result.json",
            calculator=MaceCalc(calculator_type="mace_polar", multiplicity=1),
            driver=driver,
        )
    )
    assert result["status"] == "success", result
    assert not (tmp_path / obsolete).exists()
    assert all((tmp_path / name).read_bytes() != b"previous calculation" for name in replacements)
    assert all((tmp_path / name).read_bytes() == b"previous calculation" for name in unrelated)
    rows = (tmp_path / f"frequencies_{stem}.csv").read_text().splitlines()
    assert [row.split(",")[0] for row in rows] == [
        f"{stem}_vib.{i}.traj" for i in (6, 7, 8)
    ]
    for index in (6, 7, 8):
        assert read(tmp_path / f"{stem}_vib.{index}.traj").info["mode_index"] == index


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

    calculator, vibration_runs = _controlled_spectrum(monkeypatch, energies)
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


@pytest.mark.parametrize(
    "energies,expected_indices",
    [
        pytest.param(
            [
                0.00748031j,
                0.00398170j,
                3.96854e-8j,
                3.96839e-8j,
                3.12353e-10j,
                8.62908e-5,
                9.16070e-5,
                9.16070e-5,
                0.0234627,
                0.0297504,
                0.0297504,
                0.0352987,
            ],
            [6, 7, 8, 9, 10, 11],
            id="recorded-cu4",
        ),
        pytest.param([0.01] * 7 + [0.02, 0.03], [6, 7, 8], id="boundary-duplicates"),
        pytest.param([0.03, 0.01, 0.02] + [0.001] * 6, [1, 2, 0], id="unordered"),
    ],
)
def test_thermo_reports_exactly_the_modes_used_by_ase(
    tmp_path,
    monkeypatch,
    caplog,
    energies,
    expected_indices,
):
    _controlled_spectrum(monkeypatch, energies)
    atoms = (
        Atoms("Cu4", positions=[[0, 0, 0], [2.3, 0, 0], [2.3, 2.3, 0], [0, 2.3, 0]])
        if len(energies) == 12
        else Atoms("OH2", positions=[[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]])
    )
    output = _run_thermo(
        tmp_path,
        monkeypatch,
        atoms,
        MaceCalc(calculator_type="mace_polar", multiplicity=1),
    )
    reference = IdealGasThermo(
        vib_energies=energies,
        geometry="nonlinear",
        atoms=atoms,
        potentialenergy=-1.0,
        spin=0,
        symmetrynumber=get_symmetry_number(
            AtomsData(
                numbers=atoms.numbers,
                positions=atoms.positions,
            )
        ),
        vib_selection="highest",
        ignore_imag_modes=False,
    )
    thermo, vibration = output["thermochemistry"], output["vibrational_frequencies"]
    _assert_reference(thermo, reference, 298.15, 101325)
    assert vibration["mode_indices"] == expected_indices
    assert np.array(vibration["energies"], dtype=float) / 1e3 == pytest.approx(
        reference.vib_energies
    )
    assert np.array(
        vibration["frequencies"], dtype=float
    ) * units.invcm == pytest.approx(reference.vib_energies)
    assert len(vibration["all_modes"]) == len(energies)
    for i, (original, record) in enumerate(zip(energies, vibration["all_modes"])):
        assert record["mode_index"] == i
        assert record["energy"].endswith("i") == bool(np.iscomplex(original))
        magnitude = (
            complex(original).imag if np.iscomplex(original) else float(original)
        )
        assert float(record["energy"].removesuffix("i")) / 1e3 == pytest.approx(
            magnitude
        )
    assert thermo["raw_imaginary_mode_count"] == np.count_nonzero(
        np.iscomplex(energies)
    )
    assert thermo["vib_selection"] == "highest"
    assert thermo["ase_version"] == ase.__version__
    assert thermo["warnings"] == []
    assert "The input spectrum contains" not in caplog.text
    rows = (tmp_path / "frequencies_input.csv").read_text(encoding="utf-8").splitlines()
    assert rows == [
        f"input_vib.{i}.traj,{frequency}"
        for i, frequency in zip(expected_indices, vibration["frequencies"])
    ]
    assert {path.name for path in tmp_path.glob("input_vib.*.traj")} == {
        f"input_vib.{i}.traj" for i in expected_indices
    }
    for index in expected_indices:
        assert read(tmp_path / f"input_vib.{index}.traj").info["mode_index"] == index


@pytest.mark.parametrize(
    "failure",
    [
        "constructor", "entropy", "nonfinite", "remaining-imaginary",
        "all-imaginary", "selected-zero", "all-zero",
    ],
)
def test_thermo_failure_preserves_completed_results(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    energies = {
        "remaining-imaginary": [
            0.09j, 0.08j, 0.07j, 0.06j, 0.05j, 0.04j, 0.03j, 0.02, 0.03,
        ],
        "all-imaginary": [0.01j] * 9,
        "selected-zero": [0.01j] * 6 + [0, 0.02, 0.03],
        "all-zero": [0.0] * 9,
    }.get(failure, [0.01j] * 6 + [0.02, 0.03, 0.04])
    _controlled_spectrum(monkeypatch, energies)
    if failure == "constructor":
        monkeypatch.setattr(
            "ase.thermochemistry.IdealGasThermo",
            Mock(side_effect=ValueError("thermo unavailable")),
        )
    elif failure == "entropy":
        monkeypatch.setattr(
            IdealGasThermo,
            "get_entropy",
            Mock(side_effect=ValueError("thermo unavailable")),
        )
    elif failure == "nonfinite":
        monkeypatch.setattr(
            IdealGasThermo, "get_entropy", Mock(return_value=float("nan"))
        )
    if failure in {"constructor", "entropy"}:
        expected_error = "thermo unavailable"
    elif failure in {"remaining-imaginary", "all-imaginary"}:
        expected_error = "Imaginary vibrational energies are present."
    else:
        expected_error = "ASE returned non-finite thermochemistry values."
    atoms = Atoms("OH2", positions=[[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]])
    write(tmp_path / "input.xyz", atoms)
    result = run_ase_core(
        ASEInputSchema(
            input_structure_file="input.xyz",
            output_results_file="result.json",
            calculator=MaceCalc(calculator_type="mace_polar", multiplicity=1),
            driver="thermo",
        )
    )
    assert result["status"] == "failure"
    assert result["error_type"] == "ValueError"
    assert expected_error in result["message"]
    assert "result" not in result
    assert result["converged"] is True
    assert result["potential_energy"] == -1.0
    assert result["results_file"] == str(tmp_path / "result.json")
    json.dumps(result)  # The failure payload must also work over MCP/ensemble JSON.
    output = json.loads(Path(result["results_file"]).read_text(encoding="utf-8"))
    assert output["success"] is False
    assert output["error"] == expected_error
    assert output["thermochemistry"] == {}
    assert output["final_structure"]["numbers"] == atoms.numbers.tolist()
    assert output["potential_energy"] == -1.0
    assert len(output["vibrational_frequencies"]["all_modes"]) == 9
    assert [
        complex(mode["energy"].replace("i", "j")) / 1e3
        for mode in output["vibrational_frequencies"]["all_modes"]
    ] == pytest.approx(energies)
    assert output["vibrational_frequencies"]["mode_indices"] == []
    assert read(tmp_path / "input_opt.traj").get_potential_energy() == -1.0
