from pathlib import Path
import json
import pytest
from chemgraph.tools.ase_tools import (
    run_ase,
    get_symmetry_number,
    is_linear_molecule,
)
from chemgraph.tools.cheminformatics_tools import (
    smiles_to_atomsdata,
    molecule_name_to_smiles,
)
from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.ase_input import ASEInputSchema

TEST_DIR = Path(__file__).parent


class FakeCompound:
    def __init__(self, connectivity_smiles, smiles=None):
        self.connectivity_smiles = connectivity_smiles
        self.smiles = smiles if smiles is not None else connectivity_smiles


def _patch_pubchem(monkeypatch, lookup):
    def fake_get_compounds(name, namespace):
        assert namespace == "name"
        if name in lookup:
            return [lookup[name]]
        return []

    import chemgraph.tools.cheminformatics_core as _core

    monkeypatch.setattr(_core.pcp, "get_compounds", fake_get_compounds)


def test_molecule_name_to_smiles(monkeypatch):
    _patch_pubchem(
        monkeypatch,
        {"water": FakeCompound("O"), "methane": FakeCompound("C")},
    )

    # Test with a known molecule
    assert molecule_name_to_smiles.invoke("water")['smiles'] == "O"
    assert molecule_name_to_smiles.invoke("methane")['smiles'] == "C"

    # Test with invalid molecule name
    with pytest.raises(Exception):
        molecule_name_to_smiles.invoke("not_a_real_molecule_name")


def test_molecule_name_to_smiles_stereochemistry(monkeypatch):
    """The stereochemistry flag picks the isomeric SMILES, and defaults off."""
    _patch_pubchem(
        monkeypatch,
        {"L-alanine": FakeCompound("CC(C(=O)O)N", "C[C@@H](C(=O)O)N")},
    )

    stripped = molecule_name_to_smiles.invoke("L-alanine")["smiles"]
    assert stripped == "CC(C(=O)O)N"

    kept = molecule_name_to_smiles.invoke(
        {"name": "L-alanine", "include_stereochemistry": True}
    )["smiles"]
    assert kept == "C[C@@H](C(=O)O)N"


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
def co2_atomsdata():
    """Fixture for CO2 atomsdata"""
    numbers = [6, 8, 8]
    positions = [[0, 0, 0], [1.16, 0, 0], [-1.16, 0, 0]]
    atomsdata_input = {"numbers": numbers, "positions": positions}
    return AtomsData(**atomsdata_input)


def test_get_symmetry_number(water_atomsdata):
    """Test get_symmetry_number function."""
    symmetrynumber = get_symmetry_number.invoke({"atomsdata": water_atomsdata})
    assert isinstance(symmetrynumber, int)


def test_is_linear_molecule(water_atomsdata, co2_atomsdata):
    """Test is_linear_molecule function."""
    islinear_water = is_linear_molecule.invoke({"atomsdata": water_atomsdata})
    islinear_co2 = is_linear_molecule.invoke({"atomsdata": co2_atomsdata})
    assert not islinear_water
    assert islinear_co2


@pytest.mark.parametrize("driver", ["energy", "opt", "vib", "thermo"])
def test_run_ase_drivers(monkeypatch, tmp_path, driver):
    """Exercise driver results and artifacts without downloading a model."""
    from ase import Atoms
    from ase.io import write

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    structure = tmp_path / "hydrogen.xyz"
    write(structure, Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.75]]))
    output = tmp_path / "result.json"
    params = ASEInputSchema(
        input_structure_file=str(structure),
        output_results_file=str(output),
        driver=driver,
        calculator={"calculator_type": "emt"},
        optimizer="bfgs",
        fmax=0.01,
        steps=50,
        temperature=298,
    )

    result = run_ase.invoke({"params": params})

    assert result["status"] == "success", result
    assert result["driver"] == driver
    assert isinstance(result["potential_energy"], float)
    assert result["energy_unit"] == "eV"
    assert Path(result["results_file"]) == output
    data = json.loads(output.read_text())
    assert data["success"] is True
    assert data["simulation_input"]["driver"] == driver
    assert data["potential_energy"] == result["potential_energy"]
    assert data["single_point_energy"] == result["potential_energy"]

    if driver in {"energy", "opt"}:
        assert result["single_point_energy"] == result["potential_energy"]
        assert result["unit"] == "eV"
    if driver != "energy":
        assert data["converged"] is True
    if driver == "opt":
        assert Path(result["trajectory_file"]).is_file()
    if driver in {"vib", "thermo"}:
        vibrations = data["vibrational_frequencies"]
        assert len(vibrations["energies"]) == 1
        assert len(vibrations["frequencies"]) == 1
        assert list(tmp_path.glob("hydrogen_vib.*.traj"))
        assert (tmp_path / "frequencies_hydrogen.csv").is_file()
    if driver == "thermo":
        thermo = data["thermochemistry"]
        assert {"enthalpy", "entropy", "gibbs_free_energy", "unit"} <= thermo.keys()
        assert result["result"]["thermochemistry"] == thermo


def test_run_ase_opt_writes_trajectory_next_to_output(tmp_path):
    """The optimizer records its path as {stem}_opt.traj beside the output."""
    import shutil

    input_xyz = tmp_path / "water.xyz"
    shutil.copy2(TEST_DIR / "water.xyz", input_xyz)
    schema = ASEInputSchema(
        input_structure_file=str(input_xyz),
        output_results_file=str(tmp_path / "water_output.json"),
        driver="opt",
        optimizer="bfgs",
        fmax=0.5,
        steps=5,
        calculator={"calculator_type": "emt"},
    )

    result = run_ase.invoke({"params": schema})

    assert result["status"] == "success"
    traj_path = Path(result["trajectory_file"])
    assert traj_path == tmp_path / "water_opt.traj"
    assert traj_path.exists()

    from ase.io.trajectory import Trajectory

    with Trajectory(str(traj_path)) as traj:
        energies = [float(a.get_potential_energy()) for a in traj]
    assert len(energies) >= 1


def test_write_ir_spectrum_csv_roundtrip(tmp_path):
    from chemgraph.tools.ase_core import _write_ir_spectrum_csv

    path = tmp_path / "ir_spectrum_water.csv"
    _write_ir_spectrum_csv(str(path), [500.0, 1500.25], [0.0, 1.5e-3])

    lines = path.read_text().splitlines()
    assert lines[0] == "frequency_cm1,intensity"
    assert lines[1] == "500.0000,0"
    assert lines[2] == "1500.2500,0.0015"


def test_write_ir_peaks_csv_roundtrip(tmp_path):
    from chemgraph.tools.ase_core import _write_ir_peaks_csv

    path = tmp_path / "ir_peaks_water.csv"
    _write_ir_peaks_csv(
        str(path),
        [(0, "45.1200i", 0.001), (6, "1595.4321", 1.25)],
    )

    lines = path.read_text().splitlines()
    assert lines[0] == "mode,frequency_cm1,intensity"
    assert lines[1] == "0,45.1200i,0.001"
    assert lines[2] == "6,1595.4321,1.25"
