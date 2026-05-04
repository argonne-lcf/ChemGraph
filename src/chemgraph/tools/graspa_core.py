"""Pure-Python gRASPA simulation helpers (no LangChain / MCP decorators).

Contains the core workflow functions for running gRASPA-SYCL
simulations, parsing output, and mock simulations for testing.
Used by the LangChain ``@tool`` wrapper in :mod:`graspa_tools` and the
MCP/Parsl wrappers in :mod:`chemgraph.mcp.graspa_mcp_parsl`.
"""

from __future__ import annotations

import glob
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

import ase
import numpy as np
from ase.io import read as ase_read

from chemgraph.schemas.graspa_schema import graspa_input_schema

# Template directory for gRASPA-SYCL input files
_file_dir = Path(__file__).parent / "files" / "template_graspa_sycl"

# gRASPA-SYCL command
graspa_cmd = (
    "export OMP_NUM_THREADS=1; "
    "export ZE_FLAT_DEVICE_HIERARCHY=FLAT; "
    "/lus/flare/projects/IQC/thang/soft/gRASPA/graspa-sycl/bin/sycl.out"
)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------


def _read_graspa_sycl_output(
    output_path: str,
    adsorbate: str = "H2O",
    cifname: str = None,
    output_fname: str = "raspa.log",
    temperature: float = None,
    pressure: float = None,
) -> dict:
    """Parse gRASPA output and return uptake results.

    Parameters
    ----------
    output_path : str
        Directory containing the gRASPA output files.
    adsorbate : str
        Name of the adsorbate molecule.
    cifname : str, optional
        Stem name of the CIF file (without extension).
    output_fname : str
        Name of the gRASPA log file.
    temperature : float, optional
        Simulation temperature in Kelvin.
    pressure : float, optional
        Simulation pressure in Pascal.

    Returns
    -------
    dict
        Parsed results including uptake, status, and CIF path.
    """
    result = {
        "status": "failure",
        "uptake_in_mol_kg": 0,
        "adsorbate": adsorbate,
        "temperature_in_K": None,
        "pressure_in_Pa": None,
        "cif_path": None,
    }

    target_file = Path(output_path) / Path(output_fname).name

    # --- Resolve CIF Path ---
    if cifname is None:
        cif_list = glob.glob(os.path.join(output_path, "*.cif"))
        if len(cif_list) != 1:
            cifpath = None
        else:
            cifpath = os.path.abspath(cif_list[0])
    else:
        cifpath = os.path.abspath(os.path.join(output_path, f"{cifname}.cif"))

    result["cif_path"] = cifpath

    # --- Check Log Existence ---
    if not os.path.exists(target_file):
        return result

    # --- Parse Log ---
    unitcell_line = None
    uptake_line = None

    with open(target_file, "r") as rf:
        for line in rf:
            if "UnitCells" in line:
                unitcell_line = line.strip()
            elif "Overall: Average:" in line:
                uptake_line = line.strip()

    if unitcell_line is None or uptake_line is None:
        return result

    try:
        if cifpath is None:
            raise ValueError(f"Could not resolve CIF path in {output_path}")

        uptake_total_molecule = float(uptake_line.split()[2][:-1])
        unitcell = unitcell_line.split()[4:]
        unitcell = [int(float(i)) for i in unitcell]

        atoms = ase_read(cifpath)
        framework_mass = (
            sum(atoms.get_masses()) * unitcell[0] * unitcell[1] * unitcell[2]
        )

        uptake_mol_kg = round((uptake_total_molecule / framework_mass) * 1000, 2)
        result["uptake_in_mol_kg"] = float(uptake_mol_kg)
        result["status"] = "success"
        result["temperature_in_K"] = temperature
        result["pressure_in_Pa"] = pressure
    except Exception as e:
        print(f"Error parsing results in {output_path}: {e}")
        result["status"] = "failure"

    return result


# ---------------------------------------------------------------------------
# Mock simulation (for testing)
# ---------------------------------------------------------------------------


def mock_graspa(params: graspa_input_schema) -> dict:
    """Return mock gRASPA results for testing without the SYCL runtime.

    Parameters
    ----------
    params : graspa_input_schema
        Input parameters (only ``adsorbates`` is used to determine output shape).

    Returns
    -------
    dict
        Simulated uptake results.
    """

    def rand_uptake(
        low: float, high: float, ndigits: int = 3, min_positive: float | None = None
    ) -> float:
        value = random.uniform(low, high)
        value = round(value, ndigits)
        if min_positive is not None and value == 0.0:
            value = min_positive
        return value

    time.sleep(random.uniform(20, 40))
    n_ads = len(params.adsorbates)

    if n_ads == 1:
        uptake_co2 = rand_uptake(0, 2, ndigits=3)
        return {"co2_uptake_mol_per_kg": uptake_co2}

    elif n_ads == 2:
        uptake_co2 = rand_uptake(0, 2, ndigits=3)
        uptake_n2 = rand_uptake(0, 0.5, ndigits=3, min_positive=1e-3)
        try:
            selectivity = uptake_co2 / uptake_n2
        except Exception:
            selectivity = 1e4
        return {
            "co2_uptake_mol_per_kg": uptake_co2,
            "n2_uptake_mol_per_kg": uptake_n2,
            "co2_n2_selectivity": round(selectivity, 2),
        }

    elif n_ads == 3:
        uptake_co2 = rand_uptake(0, 2, ndigits=3)
        uptake_n2 = rand_uptake(0, 0.5, ndigits=3, min_positive=1e-3)
        uptake_h2o = rand_uptake(0, 5, ndigits=3)
        try:
            selectivity = uptake_co2 / uptake_n2
        except Exception:
            selectivity = 1e4
        return {
            "co2_uptake_mol_per_kg": uptake_co2,
            "n2_uptake_mol_per_kg": uptake_n2,
            "h2o_uptake_mol_per_kg": uptake_h2o,
            "co2_n2_selectivity": round(selectivity, 2),
        }

    else:
        raise ValueError("Only supports 1-3 adsorbates only.")


# ---------------------------------------------------------------------------
# Core simulation runner
# ---------------------------------------------------------------------------


def run_graspa_core(params: graspa_input_schema) -> dict:
    """Run a single gRASPA calculation using specified input parameters.

    Parameters
    ----------
    params : graspa_input_schema
        Input parameters for the gRASPA calculation.

    Returns
    -------
    dict
        Parsed simulation results including uptake and status.
    """

    def _calculate_cell_size(
        atoms: ase.Atoms, cutoff: float = 12.8
    ) -> list[int]:
        """Calculate unit-cell replication for GCMC with the given cutoff."""
        unit_cell = atoms.cell[:]
        a = unit_cell[0]
        b = unit_cell[1]
        c = unit_cell[2]

        wa = np.divide(
            np.linalg.norm(np.dot(np.cross(b, c), a)),
            np.linalg.norm(np.cross(b, c)),
        )
        wb = np.divide(
            np.linalg.norm(np.dot(np.cross(c, a), b)),
            np.linalg.norm(np.cross(c, a)),
        )
        wc = np.divide(
            np.linalg.norm(np.dot(np.cross(a, b), c)),
            np.linalg.norm(np.cross(a, b)),
        )

        uc_x = int(np.ceil(cutoff / (0.5 * wa)))
        uc_y = int(np.ceil(cutoff / (0.5 * wb)))
        uc_z = int(np.ceil(cutoff / (0.5 * wc)))

        return [uc_x, uc_y, uc_z]

    cif_path = Path(params.input_structure_file).resolve()
    if not cif_path.exists():
        raise FileNotFoundError(f"CIF file does not exist: {cif_path}")

    base_dir = cif_path.parent

    cifname = cif_path.stem
    temperature = params.temperature
    pressure = params.pressure
    adsorbate = params.adsorbate
    n_cycle = params.n_cycles

    folder_name = f"{cifname}--{adsorbate}-{temperature}-{pressure:g}"
    sim_dir = base_dir / folder_name
    sim_dir.mkdir(parents=True, exist_ok=True)

    for item in _file_dir.iterdir():
        dest = sim_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, sim_dir)

    # Copy the specific CIF file
    shutil.copy2(cif_path, sim_dir / f"{cifname}.cif")

    atoms = ase_read(cif_path)
    [uc_x, uc_y, uc_z] = _calculate_cell_size(atoms)

    input_file = sim_dir / "simulation.input"
    temp_file = sim_dir / "simulation.input.tmp"

    with open(input_file, "r") as f_in, open(temp_file, "w") as f_out:
        for line in f_in:
            if "NCYCLE" in line:
                line = line.replace("NCYCLE", str(n_cycle))
            if "ADSORBATE" in line:
                line = line.replace("ADSORBATE", adsorbate)
            if "TEMPERATURE" in line:
                line = line.replace("TEMPERATURE", str(temperature))
            if "PRESSURE" in line:
                line = line.replace("PRESSURE", str(pressure))
            if "UC_X UC_Y UC_Z" in line:
                line = line.replace("UC_X UC_Y UC_Z", f"{uc_x} {uc_y} {uc_z}")
            if "CUTOFF" in line:
                line = line.replace("CUTOFF", str(12.8))
            if "CIFFILE" in line:
                line = line.replace("CIFFILE", cifname)
            f_out.write(line)

    shutil.move(temp_file, input_file)
    output_filename = Path(params.output_result_file).name
    with (
        open(os.path.join(sim_dir, output_filename), "w") as fp,
        open(os.path.join(sim_dir, "raspa.err"), "w") as fe,
    ):
        subprocess.run(graspa_cmd, cwd=sim_dir, stdout=fp, stderr=fe, shell=True)

    return _read_graspa_sycl_output(
        output_path=str(sim_dir),
        adsorbate=adsorbate,
        cifname=cifname,
        output_fname=params.output_result_file,
        temperature=temperature,
        pressure=pressure,
    )
