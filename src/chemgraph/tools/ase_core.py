"""Core simulation functions — the single source of truth.

Every callable here is a plain Python function (no LangChain ``@tool``,
no MCP ``@mcp.tool``, no Parsl ``@python_app``).  Framework-specific
wrappers in ``ase_tools.py``, ``mcp_tools.py``, and ``parsl_tools.py``
simply delegate to these functions.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """If ``CHEMGRAPH_LOG_DIR`` is set and *path* is relative, prepend it."""
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if log_dir and not os.path.isabs(path):
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, path)
    return path


# ---------------------------------------------------------------------------
# AtomsData <-> ASE Atoms conversions
# ---------------------------------------------------------------------------

def atoms_to_atomsdata(atoms) -> AtomsData:
    """Convert an ASE ``Atoms`` object to :class:`AtomsData`.

    Parameters
    ----------
    atoms : ase.Atoms
        ASE Atoms object.

    Returns
    -------
    AtomsData
    """
    return AtomsData(
        numbers=atoms.numbers.tolist(),
        positions=atoms.positions.tolist(),
        cell=atoms.cell.tolist(),
        pbc=atoms.pbc.tolist(),
    )


def atomsdata_to_atoms(atomsdata: AtomsData):
    """Convert :class:`AtomsData` to an ASE ``Atoms`` object.

    Parameters
    ----------
    atomsdata : AtomsData

    Returns
    -------
    ase.Atoms
    """
    from ase import Atoms

    return Atoms(
        numbers=atomsdata.numbers,
        positions=atomsdata.positions,
        cell=atomsdata.cell,
        pbc=atomsdata.pbc,
    )


# ---------------------------------------------------------------------------
# Molecular property helpers
# ---------------------------------------------------------------------------

def is_linear_molecule(atomsdata: AtomsData, tol: float = 1e-3) -> bool:
    """Determine whether a molecule is linear.

    Parameters
    ----------
    atomsdata : AtomsData
        Molecular structure.
    tol : float, optional
        Tolerance for the second singular value ratio, by default 1e-3.

    Returns
    -------
    bool
        ``True`` if the molecule is linear.
    """
    coords = np.array(atomsdata.positions)
    centered = coords - np.mean(coords, axis=0)
    _, s, _ = np.linalg.svd(centered)
    if s[0] == 0:
        return False  # degenerate — all atoms at one point
    return (s[1] / s[0]) < tol


def get_symmetry_number(atomsdata: AtomsData) -> int:
    """Return the rotational symmetry number using Pymatgen.

    Parameters
    ----------
    atomsdata : AtomsData

    Returns
    -------
    int
    """
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    from ase import Atoms
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = Atoms(
        numbers=atomsdata.numbers,
        positions=atomsdata.positions,
        cell=atomsdata.cell,
        pbc=atomsdata.pbc,
    )
    aaa = AseAtomsAdaptor()
    molecule = aaa.get_molecule(atoms)
    pga = PointGroupAnalyzer(molecule)
    return pga.get_rotational_symmetry_number()


# ---------------------------------------------------------------------------
# Calculator loading
# ---------------------------------------------------------------------------

def load_calculator(calculator: dict) -> tuple[object, dict, object]:
    """Instantiate an ASE calculator from a config dictionary.

    Parameters
    ----------
    calculator : dict
        Must contain a ``"calculator_type"`` key.

    Returns
    -------
    tuple[object, dict, object]
        ``(ase_calculator, extra_info, calc_schema_instance)``

    Raises
    ------
    ValueError
        If the calculator type is unsupported.
    """
    calc_type = calculator["calculator_type"].lower()

    if "emt" in calc_type:
        from chemgraph.schemas.calculators.emt_calc import EMTCalc
        calc = EMTCalc(**calculator)
    elif "tblite" in calc_type:
        from chemgraph.schemas.calculators.tblite_calc import TBLiteCalc
        calc = TBLiteCalc(**calculator)
    elif "orca" in calc_type:
        from chemgraph.schemas.calculators.orca_calc import OrcaCalc
        calc = OrcaCalc(**calculator)
    elif "nwchem" in calc_type:
        from chemgraph.schemas.calculators.nwchem_calc import NWChemCalc
        calc = NWChemCalc(**calculator)
    elif "fairchem" in calc_type:
        from chemgraph.schemas.calculators.fairchem_calc import FAIRChemCalc
        calc = FAIRChemCalc(**calculator)
    elif "mace" in calc_type:
        from chemgraph.schemas.calculators.mace_calc import MaceCalc
        calc = MaceCalc(**calculator)
    elif "aimnet2" in calc_type:
        from chemgraph.schemas.calculators.aimnet2_calc import AIMNET2Calc
        calc = AIMNET2Calc(**calculator)
    else:
        raise ValueError(
            f"Unsupported calculator: {calculator}. "
            "Available calculators are EMT, TBLite (GFN2-xTB, GFN1-xTB), "
            "Orca, NWChem, FAIRChem, MACE, or AIMNET2."
        )

    extra_info: dict = {}
    if hasattr(calc, "get_atoms_properties"):
        extra_info = calc.get_atoms_properties()

    return calc.get_calculator(), extra_info, calc


# ---------------------------------------------------------------------------
# Misc helpers (kept for backward compat / UI)
# ---------------------------------------------------------------------------

def extract_ase_atoms_from_tool_result(tool_result: dict):
    """Extract ``(atomic_numbers, positions)`` from a tool-result dict.

    Returns ``(None, None)`` if extraction fails.
    """
    for keyset in ({"numbers", "positions"}, {"atomic_numbers", "positions"}):
        if keyset.issubset(tool_result.keys()):
            return tool_result[keyset.pop()], tool_result["positions"]

    if "atoms" in tool_result:
        atoms_data = tool_result["atoms"]
        if {"numbers", "positions"}.issubset(atoms_data):
            return atoms_data["numbers"], atoms_data["positions"]

    return None, None


def create_ase_atoms(atomic_numbers, positions):
    """Create an ASE ``Atoms`` object from atomic numbers and positions."""
    from ase import Atoms

    try:
        return Atoms(numbers=atomic_numbers, positions=positions)
    except Exception as e:
        print(f"Error creating ASE Atoms object: {e}")
        return None


def create_xyz_string(atomic_numbers, positions) -> Optional[str]:
    """Create an XYZ-format string from atomic numbers and positions."""
    from ase import Atoms

    try:
        atoms = Atoms(numbers=atomic_numbers, positions=positions)
        xyz_lines = [str(len(atoms)), "Generated by ChemGraph"]
        for symbol, pos in zip(atoms.get_chemical_symbols(), atoms.positions):
            xyz_lines.append(
                f"{symbol:2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}"
            )
        return "\n".join(xyz_lines)
    except Exception as e:
        print(f"Error creating XYZ string: {e}")
        return None


# ---------------------------------------------------------------------------
# Unified ASE simulation core
# ---------------------------------------------------------------------------

def run_ase_core(params: ASEInputSchema) -> dict:
    """Run an ASE simulation — the single implementation for all call methods.

    This function implements energy, dipole, optimisation, vibrational,
    thermochemistry, and IR calculations.  Framework-specific wrappers
    (LangChain ``@tool``, MCP ``@mcp.tool``, Parsl) delegate here.

    Parameters
    ----------
    params : ASEInputSchema
        Fully validated simulation input.

    Returns
    -------
    dict
        Minimal result payload (status, message, key numbers).
    """
    from ase.io import read
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin

    # ---- unpack params ----
    try:
        calculator = params.calculator.model_dump()
    except Exception as e:
        return {
            "status": "failure",
            "error_type": "ValidationError",
            "message": f"Missing calculator parameter for the simulation. Raised exception: {e}",
        }

    start_time = time.time()

    input_structure_file = params.input_structure_file
    output_results_file = _resolve_path(params.output_results_file)
    optimizer = params.optimizer
    fmax = params.fmax
    steps = params.steps
    driver = params.driver
    temperature = params.temperature
    pressure = params.pressure

    # ---- input validation ----
    if not os.path.isfile(input_structure_file):
        return {
            "status": "failure",
            "error_type": "FileNotFoundError",
            "message": f"Input structure file {input_structure_file} does not exist.",
        }

    if not output_results_file.endswith(".json"):
        return {
            "status": "failure",
            "error_type": "ValueError",
            "message": f"Output results file must end with '.json', got: {params.output_results_file}",
        }

    calc, system_info, calc_model = load_calculator(calculator)

    if calc is None:
        return {
            "status": "failure",
            "error_type": "ValueError",
            "message": (
                f"Unsupported calculator: {calculator}. Available calculators are "
                "MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB), NWChem and Orca"
            ),
        }

    try:
        atoms = read(input_structure_file)
    except Exception as e:
        return {
            "status": "failure",
            "error_type": type(e).__name__,
            "message": f"Cannot read {input_structure_file} using ASE. Exception from ASE: {e}",
        }

    atoms.info.update(system_info)
    atoms.calc = calc

    # ------------------------------------------------------------------
    # Driver: energy / dipole  (single-point, no optimisation)
    # ------------------------------------------------------------------
    if driver in ("energy", "dipole"):
        energy = atoms.get_potential_energy()
        final_structure = atoms_to_atomsdata(atoms)

        dipole: List[Optional[float]] = [None, None, None]
        if driver == "dipole":
            try:
                dipole = [round(x, 4) for x in atoms.get_dipole_moment()]
            except Exception:
                pass

        end_time = time.time()
        wall_time = end_time - start_time

        simulation_output = ASEOutputSchema(
            input_structure_file=input_structure_file,
            converged=True,
            final_structure=final_structure,
            simulation_input=params,
            success=True,
            dipole_value=dipole,
            single_point_energy=energy,
            wall_time=wall_time,
        )
        with open(output_results_file, "w", encoding="utf-8") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))

        if driver == "energy":
            return {
                "status": "success",
                "message": f"Simulation completed. Results saved to {os.path.abspath(output_results_file)}",
                "single_point_energy": energy,
                "unit": "eV",
            }
        else:  # dipole
            return {
                "status": "success",
                "message": f"Simulation completed. Results saved to {os.path.abspath(output_results_file)}",
                "dipole_moment": dipole,
            }

    # ------------------------------------------------------------------
    # Drivers that require optimisation: opt / vib / thermo / ir
    # ------------------------------------------------------------------
    OPTIMIZERS = {
        "bfgs": BFGS,
        "lbfgs": LBFGS,
        "gpmin": GPMin,
        "fire": FIRE,
        "mdmin": MDMin,
    }
    try:
        optimizer_class = OPTIMIZERS.get(optimizer.lower())
        if optimizer_class is None:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

        if len(atoms) > 1:
            dyn = optimizer_class(atoms)
            converged = dyn.run(fmax=fmax, steps=steps)
        else:
            converged = True

        single_point_energy = float(atoms.get_potential_energy())
        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        thermo_data: dict = {}
        vib_data: dict = {}
        ir_data: dict = {}

        # --------------------------------------------------------------
        # Vibrational / thermo / IR analysis
        # --------------------------------------------------------------
        if driver in {"vib", "thermo", "ir"}:
            from ase.vibrations import Vibrations
            from ase import units

            ir_plot_path: Optional[str] = None
            mol_stem = (
                Path(input_structure_file).stem if input_structure_file else "mol"
            )

            with tempfile.TemporaryDirectory(
                prefix=f"chemgraph_vib_{mol_stem}_"
            ) as tmpdir:
                vib_name = os.path.join(tmpdir, "vib")
                vib = Vibrations(atoms, name=vib_name)
                vib.clean()
                vib.run()

                vib_data = {
                    "energies": [],
                    "energy_unit": "meV",
                    "frequencies": [],
                    "frequency_unit": "cm-1",
                }

                energies = vib.get_energies()

                for _idx, e in enumerate(energies):
                    is_imag = abs(e.imag) > 1e-8
                    e_val = e.imag if is_imag else e.real
                    energy_meV = 1e3 * e_val
                    freq_cm1 = e_val / units.invcm
                    suffix = "i" if is_imag else ""
                    vib_data["energies"].append(f"{energy_meV}{suffix}")
                    vib_data["frequencies"].append(f"{freq_cm1}{suffix}")

                # Write frequencies CSV
                freq_file_path = _resolve_path(f"frequencies_{mol_stem}.csv")
                freq_file = Path(freq_file_path)
                if freq_file.exists():
                    freq_file.unlink()
                with freq_file.open("w", encoding="utf-8") as f:
                    for i, freq in enumerate(vib_data["frequencies"], start=0):
                        f.write(f"{mol_stem}_vib.{i}.traj,{freq}\n")

                # Write normal-mode .traj files, then copy out of tmpdir
                for i in range(len(energies)):
                    vib.write_mode(n=i, kT=units.kB * 300, nimages=30)

                traj_dest_dir = _resolve_path("")
                if traj_dest_dir:
                    os.makedirs(traj_dest_dir, exist_ok=True)
                for traj_file in glob.glob(os.path.join(tmpdir, "vib.*.traj")):
                    dest_name = f"{mol_stem}_{Path(traj_file).name}"
                    dest_path = (
                        os.path.join(traj_dest_dir, dest_name)
                        if traj_dest_dir
                        else dest_name
                    )
                    shutil.copy2(traj_file, dest_path)

                # ---- IR ----
                if driver == "ir":
                    from ase.vibrations import Infrared
                    import matplotlib

                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    ir_data["spectrum_frequencies"] = []
                    ir_data["spectrum_frequencies_units"] = "cm-1"
                    ir_data["spectrum_intensities"] = []
                    ir_data["spectrum_intensities_units"] = "D/Å^2 amu^-1"

                    ir_name = os.path.join(tmpdir, "ir")
                    ir = Infrared(atoms, name=ir_name)
                    ir.clean()
                    ir.run()

                    IR_SPECTRUM_START = 500
                    IR_SPECTRUM_END = 4000
                    freq_intensity = ir.get_spectrum(
                        start=IR_SPECTRUM_START, end=IR_SPECTRUM_END
                    )
                    fig, ax = plt.subplots()
                    ax.plot(freq_intensity[0], freq_intensity[1])
                    ax.set_xlabel("Frequency (cm⁻¹)")
                    ax.set_ylabel("Intensity (a.u.)")
                    ax.set_title("Infrared Spectrum")
                    ax.grid(True)
                    ir_plot_path = _resolve_path(f"ir_spectrum_{mol_stem}.png")
                    fig.savefig(ir_plot_path, format="png", dpi=300)
                    plt.close(fig)

                    ir_data["IR Plot"] = f"Saved to {os.path.abspath(ir_plot_path)}"
                    ir_data["Normal mode data"] = (
                        f"Normal modes saved as individual .traj files with prefix {mol_stem}_"
                    )

                # ---- Thermochemistry ----
                if driver == "thermo":
                    if len(atoms) == 1:
                        thermo_data = {
                            "enthalpy": single_point_energy,
                            "entropy": 0.0,
                            "gibbs_free_energy": single_point_energy,
                            "unit": "eV",
                        }
                    else:
                        from ase.thermochemistry import IdealGasThermo

                        linear = is_linear_molecule(final_structure)
                        geometry = "linear" if linear else "nonlinear"
                        symmetrynumber = get_symmetry_number(final_structure)

                        thermo = IdealGasThermo(
                            vib_energies=energies,
                            potentialenergy=single_point_energy,
                            atoms=atoms,
                            geometry=geometry,
                            symmetrynumber=symmetrynumber,
                            spin=0,
                        )
                        thermo_data = {
                            "enthalpy": float(
                                thermo.get_enthalpy(temperature=temperature)
                            ),
                            "entropy": float(
                                thermo.get_entropy(
                                    temperature=temperature, pressure=pressure
                                )
                            ),
                            "gibbs_free_energy": float(
                                thermo.get_gibbs_energy(
                                    temperature=temperature, pressure=pressure
                                )
                            ),
                            "unit": "eV",
                        }

        # ---- serialise full output ----
        end_time = time.time()
        wall_time = end_time - start_time

        simulation_output = ASEOutputSchema(
            input_structure_file=input_structure_file,
            converged=converged,
            final_structure=final_structure,
            simulation_input=params,
            vibrational_frequencies=vib_data,
            thermochemistry=thermo_data,
            success=True,
            ir_data=ir_data,
            single_point_energy=single_point_energy,
            wall_time=wall_time,
        )
        with open(output_results_file, "w", encoding="utf-8") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))

        # ---- minimal return payload ----
        abs_output = os.path.abspath(output_results_file)
        if driver == "opt":
            return {
                "status": "success",
                "message": f"Simulation completed. Results saved to {abs_output}",
                "single_point_energy": single_point_energy,
                "unit": "eV",
            }
        elif driver == "vib":
            return {
                "status": "success",
                "result": {"vibrational_frequencies": vib_data},
                "message": (
                    "Vibrational analysis completed; frequencies returned. "
                    f"Full results (structure, vibrations and metadata) saved to {abs_output}."
                ),
            }
        elif driver == "thermo":
            return {
                "status": "success",
                "result": {"thermochemistry": thermo_data},
                "message": (
                    "Thermochemistry computed and returned. "
                    f"Full results (structure, vibrations, thermochemistry and metadata) saved to {abs_output}"
                ),
            }
        elif driver == "ir":
            return {
                "status": "success",
                "result": {"vibrational_frequencies": vib_data},
                "message": (
                    "Infrared computed and returned. "
                    f"Full results (structure, vibrations, thermochemistry and metadata) saved to {abs_output}. "
                    f"IR plot saved to {os.path.abspath(ir_plot_path) if ir_plot_path else 'N/A'}. "
                    "Normal modes saved as individual .traj files"
                ),
            }

    except Exception as e:
        return {
            "status": "failure",
            "error_type": type(e).__name__,
            "message": str(e),
        }


# ---------------------------------------------------------------------------
# JSON result loader
# ---------------------------------------------------------------------------


def extract_output_json_core(json_file: str) -> dict:
    """Load simulation results from a JSON file produced by ``run_ase_core``.

    Parameters
    ----------
    json_file : str
        Path to the JSON file containing ASE simulation results.

    Returns
    -------
    dict
        Parsed results from the JSON file as a Python dictionary.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
