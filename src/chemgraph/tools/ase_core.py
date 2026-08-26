"""Core simulation functions — the single source of truth.

Every callable here is a plain Python function (no LangChain ``@tool``,
no MCP ``@mcp.tool``, no Parsl ``@python_app``).  Framework-specific
wrappers in ``ase_tools.py``, ``mcp_tools.py``, and ``parsl_tools.py``
simply delegate to these functions.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema
from chemgraph.schemas.calculators.mace_calc import MaceCalc

logger = logging.getLogger(__name__)


def _ensure_ase_core_file_log() -> None:
    """Attach a single ``FileHandler`` to the ase_core logger.

    ``run_ase_core`` runs both in the MCP-server process (where
    ``server_utils`` already configures root logging) and in worker
    processes (Parsl / EnsembleLauncher / Globus Compute) that never go
    through that setup, so we add our own file handler here. Idempotent:
    a second call is a no-op, which avoids accumulating one open file
    handle per invocation. Honors ``CHEMGRAPH_LOG_DIR`` when set.
    """
    if any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        return
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR", os.path.join(os.getcwd(), "cg_logs"))
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "ase_core.log"))
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: str) -> str:
    """Resolve a path relative to ``CHEMGRAPH_LOG_DIR`` when appropriate.

    Parameters
    ----------
    path : str
        Absolute or relative file path.

    Returns
    -------
    str
        Resolved path.
    """
    log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if log_dir and not os.path.isabs(path):
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, path)
    return path


def _resolve_existing_path(path: str) -> str:
    """Resolve a path to read that a sibling tool may have written to the log dir.

    Tools that *write* files (``smiles_to_coordinate_file``, ``run_ase``'s
    result JSON, ``save_atomsdata_to_file`` ...) send relative paths through
    :func:`_resolve_path`, so a bare ``"water.xyz"`` lands in
    ``CHEMGRAPH_LOG_DIR`` rather than the caller's cwd. A tool that later
    *reads* that bare name must look in the same place, otherwise it raises
    ``FileNotFoundError`` even though the file exists.

    This helper returns ``path`` unchanged when it already points at an
    existing file (absolute paths and genuine cwd-relative paths keep working);
    only when the raw path is missing does it fall back to the
    ``CHEMGRAPH_LOG_DIR``-resolved location. The raw path is returned when
    neither exists, so callers still surface a meaningful "not found" error.

    Parameters
    ----------
    path : str
        Absolute or relative file path to read.

    Returns
    -------
    str
        The raw path if it exists, else the log-dir-resolved path if that
        exists, else the raw path unchanged.
    """
    if os.path.isfile(path):
        return path
    resolved = _resolve_path(path)
    if resolved != path and os.path.isfile(resolved):
        return resolved
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


def _vibrational_mode_indices(atomsdata: AtomsData, total_modes: int) -> list[int]:
    """Return ASE mode indices corresponding to molecular vibrations.

    ASE returns all ``3N`` normal modes in ascending order.  The leading
    translational and rotational modes are excluded from reported vibration
    data: five modes for a linear molecule and six for a nonlinear molecule.

    Parameters
    ----------
    atomsdata : AtomsData
        Optimized molecular structure.
    total_modes : int
        Number of modes returned by ASE.

    Returns
    -------
    list[int]
        Indices of the modes to report.
    """
    num_atoms = len(atomsdata.numbers)
    expected_modes = 3 * num_atoms
    if total_modes != expected_modes:
        raise ValueError(
            f"Expected {expected_modes} normal modes for {num_atoms} atoms, "
            f"got {total_modes}."
        )

    if num_atoms == 1:
        return []

    num_nonvibrational = 5 if is_linear_molecule(atomsdata) else 6
    return list(range(num_nonvibrational, total_modes))


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
    elif "tblite" in calc_type or "xtb" in calc_type:
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

    if "mace" in calc_type:
        # MACE's torch.load + symbolic_trace is unsafe under concurrent loads,
        # whether the concurrency is threads in one process or sibling processes
        # spawned by the EnsembleLauncher process pool. See mace_calc._mace_lock.
        from chemgraph.schemas.calculators.mace_calc import mace_loading_lock

        with mace_loading_lock():
            ase_calculator = calc.get_calculator()
    else:
        ase_calculator = calc.get_calculator()

    return ase_calculator, extra_info, calc


def _simulation_input_for_output(
    params: ASEInputSchema, calc_model: object
) -> ASEInputSchema:
    """Return simulation input enriched with output-only calculator metadata."""
    if not isinstance(calc_model, MaceCalc) or calc_model.model is not None:
        return params

    model_name = calc_model.get_model_name_for_output()
    if model_name is None:
        return params

    output_calculator = calc_model.model_copy(update={"model": model_name})
    return params.model_copy(update={"calculator": output_calculator})


# ---------------------------------------------------------------------------
# Misc helpers (kept for backward compat / UI)
# ---------------------------------------------------------------------------

def extract_ase_atoms_from_tool_result(tool_result: dict):
    """Extract ``(atomic_numbers, positions)`` from a tool-result dict.

    Returns ``(None, None)`` if extraction fails.

    Parameters
    ----------
    tool_result : dict
        Tool result that may contain atom numbers and positions.

    Returns
    -------
    tuple
        ``(atomic_numbers, positions)`` or ``(None, None)``.
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
    """Create an ASE ``Atoms`` object from atomic numbers and positions.

    Parameters
    ----------
    atomic_numbers : sequence
        Atomic numbers for each atom.
    positions : sequence
        Cartesian coordinates for each atom.

    Returns
    -------
    ase.Atoms or None
        Constructed atoms object, or ``None`` if construction fails.
    """
    from ase import Atoms

    try:
        return Atoms(numbers=atomic_numbers, positions=positions)
    except Exception as e:
        print(f"Error creating ASE Atoms object: {e}")
        return None


def create_xyz_string(atomic_numbers, positions) -> Optional[str]:
    """Create an XYZ-format string from atomic numbers and positions.

    Parameters
    ----------
    atomic_numbers : sequence
        Atomic numbers for each atom.
    positions : sequence
        Cartesian coordinates for each atom.

    Returns
    -------
    str or None
        XYZ-format structure text, or ``None`` if conversion fails.
    """
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


def _energy_result_metadata(
    driver: str,
    potential_energy: float,
    results_file: str,
    *,
    converged: Optional[bool] = None,
    optimization_steps: Optional[int] = None,
) -> dict:
    """Build consistent energy metadata for a successful tool result."""
    result = {
        "driver": driver,
        "potential_energy": potential_energy,
        "energy_unit": "eV",
        "results_file": os.path.abspath(results_file),
    }
    if converged is not None:
        result["converged"] = converged
    if optimization_steps is not None:
        result["optimization_steps"] = optimization_steps
    return result


# ---------------------------------------------------------------------------
# Unified ASE simulation core
# ---------------------------------------------------------------------------

def run_ase_core(params: ASEInputSchema) -> dict:
    """Run an ASE simulation — the single implementation for all call methods.

    This function implements energy, dipole, optimization, vibrational,
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
    calculator_type = str(
        getattr(params.calculator, "calculator_type", "")
    ).lower()

    try:
        # The macOS TBLite and PyTorch wheels each bundle libomp.dylib. Loading
        # both copies in one interpreter raises SIGABRT, which no Python
        # try/except can catch. Keep TBLite in a fresh interpreter so a native
        # loader failure is contained and reported to the caller.
        if calculator_type == "tblite":
            return _run_ase_core_isolated(params)
        return _run_ase_core_in_process(params)
    except Exception as exc:
        logger.exception("run_ase_core failed with %s: %s", type(exc).__name__, exc)
        return {
            "status": "failure",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _run_ase_core_isolated(params: ASEInputSchema) -> dict:
    """Run a native calculator in a child process and contain hard crashes."""
    env = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        os.pathsep.join((source_root, existing_pythonpath))
        if existing_pythonpath
        else source_root
    )

    with tempfile.TemporaryDirectory(prefix="chemgraph_ase_worker_") as tmpdir:
        env.setdefault("MPLCONFIGDIR", os.path.join(tmpdir, "matplotlib"))
        env.setdefault("XDG_CACHE_HOME", os.path.join(tmpdir, "cache"))
        result_path = Path(tmpdir) / "result.json"
        command = [
            sys.executable,
            "-m",
            "chemgraph.tools._ase_worker",
            str(result_path),
        ]
        completed = subprocess.run(
            command,
            input=params.model_dump_json(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            check=False,
        )

        if completed.returncode != 0:
            if completed.returncode < 0:
                signal_number = -completed.returncode
                try:
                    signal_name = signal.Signals(signal_number).name
                except ValueError:
                    signal_name = f"signal {signal_number}"
                termination = f"terminated by {signal_name}"
            else:
                termination = f"exited with status {completed.returncode}"

            calculator_name = getattr(
                params.calculator, "calculator_type", "native calculator"
            )
            message = f"{calculator_name} worker {termination}."
            if "OMP: Error #15" in completed.stderr or "libomp" in completed.stderr:
                message += (
                    " A conflicting OpenMP runtime was detected and contained; "
                    "the ChemGraph process is still running."
                )
            logger.error(
                "%s Worker stderr: %s",
                message,
                completed.stderr.strip() or "<empty>",
            )
            return {
                "status": "failure",
                "error_type": "CalculatorProcessError",
                "message": message,
                "returncode": completed.returncode,
            }

        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Calculator worker returned no valid result: %s", exc)
            return {
                "status": "failure",
                "error_type": "CalculatorProcessError",
                "message": f"Calculator worker returned no valid result: {exc}",
            }

        if not isinstance(result, dict):
            return {
                "status": "failure",
                "error_type": "CalculatorProcessError",
                "message": "Calculator worker returned a non-object result.",
            }
        return result


def _run_ase_core_in_process(params: ASEInputSchema) -> dict:
    """Execute the ASE simulation in the current interpreter."""
    from ase.io import read
    from ase.optimize import BFGS, LBFGS, GPMin, FIRE, MDMin

    # ---- file logger (cg_logs/) ----
    _ensure_ase_core_file_log()

    logger.info("run_ase_core called with params: %s", params.model_dump_json())

    # ---- unpack params ----
    try:
        calculator = params.calculator.model_dump()
    except Exception as e:
        logger.error("Calculator validation failed: %s", e)
        return {
            "status": "failure",
            "error_type": "ValidationError",
            "message": f"Missing calculator parameter for the simulation. Raised exception: {e}",
        }

    start_time = time.time()

    # Resolve a relative input path against CHEMGRAPH_LOG_DIR, matching how
    # smiles_to_coordinate_file writes it. Without this, a tool that writes
    # water.xyz into the session log dir and a later run_ase that reads
    # "water.xyz" from cwd disagree -> FileNotFoundError.
    input_structure_file = _resolve_existing_path(params.input_structure_file)
    output_results_file = _resolve_path(params.output_results_file)
    optimizer = params.optimizer
    fmax = params.fmax
    steps = params.steps
    driver = params.driver
    temperature = params.temperature
    pressure = params.pressure

    # ---- input validation ----
    logger.info("driver=%s, input=%s, output=%s, optimizer=%s, fmax=%s, steps=%s",
                driver, input_structure_file, output_results_file, optimizer, fmax, steps)

    if not os.path.isfile(input_structure_file):
        logger.error("Input file not found: %s", input_structure_file)
        return {
            "status": "failure",
            "error_type": "FileNotFoundError",
            "message": f"Input structure file {input_structure_file} does not exist.",
        }

    if not output_results_file.endswith(".json"):
        logger.error("Invalid output file extension: %s", output_results_file)
        return {
            "status": "failure",
            "error_type": "ValueError",
            "message": f"Output results file must end with '.json', got: {params.output_results_file}",
        }

    # Make sure the destination directory exists before the simulation runs;
    # otherwise the trailing ``open(output_results_file, "w")`` fails with
    # FileNotFoundError after the calculation has already burned its
    # compute time. Callers (LLM agents, scripts) routinely point at a
    # not-yet-created subdirectory of a shared run dir, so create it now.
    output_parent = os.path.dirname(os.path.abspath(output_results_file))
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    logger.info("Loading calculator: %s", calculator)
    calc, system_info, calc_model = load_calculator(calculator)

    if calc is None:
        logger.error("Unsupported calculator: %s", calculator)
        return {
            "status": "failure",
            "error_type": "ValueError",
            "message": (
                f"Unsupported calculator: {calculator}. Available calculators are "
                "MACE (mace_mp, mace_off, mace_anicc), EMT, TBLite (GFN2-xTB, GFN1-xTB), NWChem and Orca"
            ),
        }
    logger.info("Calculator loaded successfully: %s", type(calc).__name__)
    simulation_input = _simulation_input_for_output(params, calc_model)

    if driver == "ir" and "dipole" not in getattr(
        calc, "implemented_properties", ()
    ):
        return {
            "status": "failure",
            "error_type": "PropertyNotImplementedError",
            "message": (
                "IR calculations require a calculator that implements dipole "
                f"moments; {type(calc).__name__} does not provide the dipole property."
            ),
        }

    try:
        atoms = read(input_structure_file)
    except Exception as e:
        logger.error("Failed to read input structure: %s", e)
        return {
            "status": "failure",
            "error_type": type(e).__name__,
            "message": f"Cannot read {input_structure_file} using ASE. Exception from ASE: {e}",
        }

    logger.info("Read %d atoms from %s", len(atoms), input_structure_file)
    atoms.info.update(system_info)
    atoms.calc = calc

    # ------------------------------------------------------------------
    # Driver: energy / dipole  (single-point, no optimization)
    # ------------------------------------------------------------------
    if driver in ("energy", "dipole"):
        logger.info("Running single-point %s calculation", driver)
        potential_energy = float(atoms.get_potential_energy())
        logger.info("Single-point energy: %s eV", potential_energy)
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
            simulation_input=simulation_input,
            success=True,
            dipole_value=dipole,
            potential_energy=potential_energy,
            single_point_energy=potential_energy,
            wall_time=wall_time,
        )
        with open(output_results_file, "w", encoding="utf-8") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))
        logger.info("Results saved to %s (wall_time=%.2fs)", output_results_file, wall_time)

        if driver == "energy":
            return {
                "status": "success",
                "message": (
                    "Single-point energy calculation completed. "
                    f"Results saved to {os.path.abspath(output_results_file)}"
                ),
                **_energy_result_metadata(
                    driver, potential_energy, output_results_file
                ),
                "single_point_energy": potential_energy,
                "unit": "eV",
            }
        else:  # dipole
            return {
                "status": "success",
                "message": (
                    "Dipole calculation completed. "
                    f"Results saved to {os.path.abspath(output_results_file)}"
                ),
                **_energy_result_metadata(
                    driver, potential_energy, output_results_file
                ),
                "dipole_moment": dipole,
                "dipole_unit": "e * Angstrom",
            }

    # ------------------------------------------------------------------
    # Drivers that require optimization: opt / vib / thermo / ir
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

        logger.info("Running optimization with %s (fmax=%s, steps=%s)", optimizer, fmax, steps)
        optimization_steps = 0
        if len(atoms) > 1:
            dyn = optimizer_class(atoms)
            converged = dyn.run(fmax=fmax, steps=steps)
            optimization_steps = dyn.nsteps
        else:
            converged = True
        logger.info("Optimization converged=%s", converged)

        potential_energy = float(atoms.get_potential_energy())
        logger.info("Post-optimization energy: %s eV", potential_energy)
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
            logger.info("Starting vibrational analysis (driver=%s)", driver)
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
                logger.info("Vibrational analysis complete")

                vib_data = {
                    "energies": [],
                    "energy_unit": "meV",
                    "frequencies": [],
                    "frequency_unit": "cm-1",
                }

                all_energies = vib.get_energies()
                mode_indices = _vibrational_mode_indices(
                    final_structure, len(all_energies)
                )

                for mode_index in mode_indices:
                    e = all_energies[mode_index]
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
                    for mode_index, freq in zip(
                        mode_indices, vib_data["frequencies"]
                    ):
                        f.write(f"{mol_stem}_vib.{mode_index}.traj,{freq}\n")

                # Write normal-mode .traj files, then copy out of tmpdir
                for mode_index in mode_indices:
                    vib.write_mode(
                        n=mode_index, kT=units.kB * 300, nimages=30
                    )

                traj_dest_dir = _resolve_path("")
                if traj_dest_dir:
                    os.makedirs(traj_dest_dir, exist_ok=True)
                stale_traj_pattern = os.path.join(
                    traj_dest_dir, f"{mol_stem}_vib.*.traj"
                )
                for stale_traj_file in glob.glob(stale_traj_pattern):
                    os.unlink(stale_traj_file)
                for mode_index in mode_indices:
                    traj_file = os.path.join(tmpdir, f"vib.{mode_index}.traj")
                    dest_name = f"{mol_stem}_{Path(traj_file).name}"
                    dest_path = (
                        os.path.join(traj_dest_dir, dest_name)
                        if traj_dest_dir
                        else dest_name
                    )
                    shutil.copy2(traj_file, dest_path)

                # ---- IR ----
                if driver == "ir":
                    logger.info("Running IR calculation")
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

                    logger.info("IR spectrum plot saved to %s", ir_plot_path)
                    ir_data["IR Plot"] = f"Saved to {os.path.abspath(ir_plot_path)}"
                    ir_data["Normal mode data"] = (
                        f"Normal modes saved as individual .traj files with prefix {mol_stem}_"
                    )

                # ---- Thermochemistry ----
                if driver == "thermo":
                    logger.info("Computing thermochemistry (T=%s K, P=%s Pa)", temperature, pressure)
                    if len(atoms) == 1:
                        thermo_data = {
                            "enthalpy": potential_energy,
                            "entropy": 0.0,
                            "gibbs_free_energy": potential_energy,
                            "unit": "eV",
                        }
                    else:
                        from ase.thermochemistry import IdealGasThermo

                        linear = is_linear_molecule(final_structure)
                        geometry = "linear" if linear else "nonlinear"
                        symmetrynumber = get_symmetry_number(final_structure)

                        # IdealGasThermo expects total spin S; calculators expose
                        # multiplicity (2S+1) via get_multiplicity() when supported.
                        multiplicity = (
                            getattr(calc_model, "get_multiplicity", lambda: None)()
                            or 1
                        )
                        spin_S = (multiplicity - 1) / 2.0

                        thermo = IdealGasThermo(
                            vib_energies=all_energies,
                            potentialenergy=potential_energy,
                            atoms=atoms,
                            geometry=geometry,
                            symmetrynumber=symmetrynumber,
                            spin=spin_S,
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
        logger.info("Simulation finished (driver=%s, wall_time=%.2fs, converged=%s)", driver, wall_time, converged)

        simulation_output = ASEOutputSchema(
            input_structure_file=input_structure_file,
            converged=converged,
            final_structure=final_structure,
            simulation_input=simulation_input,
            vibrational_frequencies=vib_data,
            thermochemistry=thermo_data,
            success=True,
            ir_data=ir_data,
            potential_energy=potential_energy,
            single_point_energy=potential_energy,
            wall_time=wall_time,
        )
        with open(output_results_file, "w", encoding="utf-8") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))

        # ---- minimal return payload ----
        abs_output = os.path.abspath(output_results_file)
        energy_metadata = _energy_result_metadata(
            driver,
            potential_energy,
            output_results_file,
            converged=converged,
            optimization_steps=optimization_steps,
        )
        if driver == "opt":
            if converged:
                message = f"Geometry optimization converged. Results saved to {abs_output}"
            else:
                message = (
                    "Geometry optimization completed without convergence after "
                    f"{optimization_steps} steps. Results saved to {abs_output}"
                )
            return {
                "status": "success",
                "message": message,
                **energy_metadata,
                "single_point_energy": potential_energy,
                "unit": "eV",
            }
        elif driver == "vib":
            return {
                "status": "success",
                **energy_metadata,
                "result": {"vibrational_frequencies": vib_data},
                "message": (
                    "Vibrational analysis completed; frequencies returned. "
                    f"Full results (structure, vibrations and metadata) saved to {abs_output}."
                ),
            }
        elif driver == "thermo":
            return {
                "status": "success",
                **energy_metadata,
                "result": {"thermochemistry": thermo_data},
                "message": (
                    "Thermochemistry computed and returned. "
                    f"Full results (structure, vibrations, thermochemistry and metadata) saved to {abs_output}"
                ),
            }
        elif driver == "ir":
            return {
                "status": "success",
                **energy_metadata,
                "result": {"vibrational_frequencies": vib_data},
                "message": (
                    "Infrared computed and returned. "
                    f"Full results (structure, vibrations, thermochemistry and metadata) saved to {abs_output}. "
                    f"IR plot saved to {os.path.abspath(ir_plot_path) if ir_plot_path else 'N/A'}. "
                    "Normal modes saved as individual .traj files"
                ),
            }

    except Exception as e:
        logger.exception("run_ase_core failed with %s: %s", type(e).__name__, e)
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
    # run_ase writes its result JSON via _resolve_path (into CHEMGRAPH_LOG_DIR),
    # so a bare relative name passed here must resolve to the same place.
    json_file = _resolve_existing_path(json_file)
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data
