import subprocess
import os
from pathlib import Path
import shutil
import random
import time
import numpy as np
import glob

import ase
from ase.io import read as ase_read

from langchain_core.tools import tool

from chemgraph.schemas.graspa_schema import (
    graspa_input_schema,
)

# LangGraph gRASPA tools.
_file_dir = Path(__file__).parent / "files" / "template_graspa_sycl"

# gRASPA-SYCL command
graspa_cmd = "export OMP_NUM_THREADS=1; export ZE_FLAT_DEVICE_HIERARCHY=FLAT; /lus/flare/projects/IQC/thang/soft/gRASPA/graspa-sycl/bin/sycl.out"


def _read_graspa_sycl_output(
    output_path: str,
    adsorbate: str = "H2O",
    cifname: str = None,
    output_fname: str = "raspa.log",
    temperature: float = None,
    pressure: float = None,
):
    """
    Parses gRASPA output and includes the full path to the CIF file used.
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
    # We resolve this early so we can return it even if the log parsing fails
    if cifname is None:
        cif_list = glob.glob(os.path.join(output_path, "*.cif"))
        if len(cif_list) != 1:
            # If explicit name not provided and we can't auto-detect unique CIF, we can't resolve path
            cifpath = None
        else:
            cifpath = os.path.abspath(cif_list[0])
    else:
        # Construct absolute path based on output_path
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
        error_total_molecule = float(uptake_line.split()[4][:-1])

        # Parse UnitCells (robust to whitespace)
        unitcell = unitcell_line.split()[4:]
        unitcell = [int(float(i)) for i in unitcell]

        atoms = ase_read(cifpath)
        framework_mass = (
            sum(atoms.get_masses()) * unitcell[0] * unitcell[1] * unitcell[2]
        )

        uptake_mol_kg = round((uptake_total_molecule / framework_mass) * 1000, 2)
        error_mol_kg = (error_total_molecule / framework_mass) * 1000

        result["uptake_in_mol_kg"] = float(uptake_mol_kg)
        # result["error_in_mol_kg"] = float(error_mol_kg)
        result["status"] = "success"
        result["temperature_in_K"] = temperature
        result["pressure_in_Pa"] = pressure
    except Exception as e:
        print(f"Error parsing results in {output_path}: {e}")
        result["status"] = "failure"

    return result


def mock_graspa(params: graspa_input_schema) -> dict:
    def rand_uptake(
        low: float, high: float, ndigits: int = 3, min_positive: float | None = None
    ) -> float:
        """Random uptake with rounding and optional minimum positive value."""
        value = random.uniform(low, high)
        value = round(value, ndigits)
        if min_positive is not None and value == 0.0:
            value = min_positive
        return value

    time.sleep(random.uniform(20, 40))
    n_ads = len(params.adsorbates)

    if n_ads == 1:
        uptake_co2 = rand_uptake(0, 2, ndigits=3)
        return {
            "co2_uptake_mol_per_kg": uptake_co2,
        }

    elif n_ads == 2:
        uptake_co2 = rand_uptake(0, 2, ndigits=3)
        # prevent rounded value from becoming exactly zero
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
        raise ValueError("Only supports 1–3 adsorbates only.")


def run_graspa_core(params: graspa_input_schema):
    """Run a single gRASPA calculations using specified input parameters.

    Parameters
    ----------
    params : graspa_input_schema
        Input parameters for the gRASPA calculation
    """

    def _calculate_cell_size(
        atoms: ase.Atoms, cutoff: float = 12.8
    ) -> list[int, int, int]:
        """Method to calculate Unitcells (for periodic boundary condition) for GCMC

        Args:
            atoms (ase.Atoms): ASE atom object
            cutoff (float, optional): Cutoff in Angstrom. Defaults to 12.8.

        Returns:
            list[int, int, int]: Unit cell in x, y and z
        """
        unit_cell = atoms.cell[:]
        # Unit cell vectors
        a = unit_cell[0]
        b = unit_cell[1]
        c = unit_cell[2]
        # minimum distances between unit cell faces
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
                line = line.replace(
                    "CUTOFF", str(12.8)
                )  # Default or params.cutoff if added
            if "CIFFILE" in line:
                line = line.replace("CIFFILE", cifname)
            f_out.write(line)

    shutil.move(temp_file, input_file)
    output_filename = Path(params.output_result_file).name
    with (
        open(os.path.join(sim_dir, output_filename), "w") as fp,
        open(os.path.join(sim_dir, "raspa.err"), "w") as fe,
    ):
        result = subprocess.run(
            graspa_cmd, cwd=sim_dir, stdout=fp, stderr=fe, shell=True
        )

    return _read_graspa_sycl_output(
        output_path=str(sim_dir),
        adsorbate=adsorbate,
        cifname=cifname,
        output_fname=params.output_result_file,
        temperature=temperature,
        pressure=pressure,
    )


@tool
def run_graspa(graspa_input: graspa_input_schema):
    """
    Run a gRASPA simulation using the core engine and return the uptakes.
    This tool acts as a wrapper for the agentic workflow.
    """
    # Map GRASPAInputSchema fields to the internal schema expected by run_graspa_core
    params = graspa_input_schema(
        input_structure_file=graspa_input.cif_path,
        adsorbate=graspa_input.adsorbate,
        temperature=graspa_input.temperature,
        pressure=graspa_input.pressure,
        n_cycles=graspa_input.n_cycle,
        output_result_file="raspa.log"
    )

    # Execute core logic
    result = run_graspa_core(params)

    # Return the parsed metrics
    # Note: run_graspa_core returns a dict; we extract what the tool usually expects
    if result["status"] == "success":
        return result["uptake_in_mol_kg"]
    else:
        raise RuntimeError(f"gRASPA simulation failed for {graspa_input.mof_name}")