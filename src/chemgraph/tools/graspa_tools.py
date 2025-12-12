import subprocess
import os
from pathlib import Path
import shutil
import random
import time
import numpy as np

import ase
from ase.io import read as ase_read

from langchain_core.tools import tool

from chemgraph.schemas.graspa_schema import (
    graspa_input_schema,
    graspa_input_schema_ensemble,
)
from chemgraph.models.graspa_input import GRASPAInputSchema


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

    return mock_graspa(params)


# LangGraph gRASPA tools.
_file_dir = Path(__file__).parent / "files" / "template"


@tool
def run_graspa(graspa_input: GRASPAInputSchema):
    """Run a gRASPA simulation and return the uptakes and errors. Only support single adsorbate.

    Args:
        graspa_input (str): a GRASPAInputSchema.

    Returns:
        Computed uptake (U) and error (E) from gRASPA-sycl in the following order:
            - U (mol/kg)
            - E (mol/kg)
            - U (g/L)
            - E (g/L)
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

    output_path = Path(graspa_input.output_path)
    mof_name = graspa_input.mof_name
    cif_path = graspa_input.cif_path
    adsorbate = graspa_input.adsorbate
    temperature = graspa_input.temperature
    pressure = graspa_input.pressure
    n_cycle = graspa_input.n_cycle
    cutoff = graspa_input.cutoff
    graspa_cmd = graspa_input.graspa_cmd
    graspa_version = graspa_input.graspa_version

    # Check cif file exists and '.cif' extension
    if not cif_path.endswith('.cif'):
        raise ValueError(f"CIF file {cif_path} does not have '.cif' extension.")
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file {cif_path} does not exist.")

    # Check if the mof_name contains ".cif"
    if mof_name.endswith('.cif'):
        mof_name = mof_name[:-4]

    # Create output directory
    out_dir = output_path / f"{mof_name}_{adsorbate}_{temperature}_{pressure:0e}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Calculate unit cell size
    atoms = ase_read(cif_path)
    [uc_x, uc_y, uc_z] = _calculate_cell_size(atoms=atoms)

    # Copy other input files (simulation.input, force fields and definition files) from template folder.
    subprocess.run(f"cp {_file_dir}/* {out_dir}/", shell=True)
    shutil.copy2(cif_path, os.path.join(out_dir, mof_name + '.cif'))

    # Modify input from template simulation.input
    with (
        open(f"{out_dir}/simulation.input", "r") as f_in,
        open(f"{out_dir}/simulation.input.tmp", "w") as f_out,
    ):
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
                line = line.replace("CUTOFF", str(cutoff))
            if "CIFFILE" in line:
                line = line.replace("CIFFILE", mof_name)
            f_out.write(line)

    shutil.move(f"{out_dir}/simulation.input.tmp", f"{out_dir}/simulation.input")

    # Run gRASPA-sycl
    subprocess.run(
        graspa_cmd,
        shell=True,
        cwd=out_dir,
    )

    # Get output from raspa.log when simulation is done

    if graspa_version == 'sycl':
        with open(f"{out_dir}/raspa.log", "r") as rf:
            for line in rf:
                if "UnitCells" in line:
                    unitcell_line = line.strip()
                elif "Overall: Average:" in line:
                    uptake_line = line.strip()

        unitcell = unitcell_line.split()[4:]
        unitcell = [int(float(i)) for i in unitcell]
        uptake_total_molecule = float(uptake_line.split()[2][:-1])
        error_total_molecule = float(uptake_line.split()[4][:-1])

        # Get unit in mol/kg
        framework_mass = sum(atoms.get_masses())
        framework_mass = framework_mass * unitcell[0] * unitcell[1] * unitcell[2]
        uptake_mol_kg = uptake_total_molecule / framework_mass * 1000
        error_mol_kg = error_total_molecule / framework_mass * 1000

        # Get unit in g/L
        framework_vol = atoms.get_volume()  # in Angstrom^3
        framework_vol_in_L = (
            framework_vol * 1e-27 * unitcell[0] * unitcell[1] * unitcell[2]
        )

        # Hard code for CO2 and H2
        if adsorbate == "CO2":
            molar_mass = 44.0098
        elif adsorbate == "H2":
            molar_mass = 2.02
        elif adsorbate == "CH4":
            molar_mass = 16.04
        elif adsorbate == "N2":
            molar_mass = 28.01
        else:
            raise ValueError(f"Adsorbate {adsorbate} is not supported.")
        uptake_g_L = (
            uptake_total_molecule / (6.022 * 1e23) * molar_mass / framework_vol_in_L
        )
        error_g_L = (
            error_total_molecule / (6.022 * 1e23) * molar_mass / framework_vol_in_L
        )
    elif graspa_version == 'cuda':
        uptake_lines = []
        with open(f"{out_dir}/raspa.log", "r") as rf:
            for line in rf:
                if "Overall: Average" in line:
                    uptake_lines.append(line.strip())

        result_mol_kg = uptake_lines[11].split(",")
        uptake_mol_kg = result_mol_kg[0].split()[-1]
        error_mol_kg = result_mol_kg[1].split()[-1]

        result_g_L = uptake_lines[13].split(",")
        uptake_g_L = result_g_L[0].split()[-1]
        error_g_L = result_g_L[1].split()[-1]
    else:
        raise ValueError(f"gRASPA version {graspa_version} is not supported.")

    return uptake_mol_kg, error_mol_kg, uptake_g_L, error_g_L
