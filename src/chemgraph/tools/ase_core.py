"""Core simulation functions — the single source of truth.

Every callable here is a plain Python function (no LangChain ``@tool``,
no MCP ``@mcp.tool``, no Parsl ``@python_app``).  Framework-specific
wrappers in ``ase_tools.py``, ``mcp_tools.py``, and ``parsl_tools.py``
simply delegate to these functions.
"""

from __future__ import annotations

import contextlib
import glob
import json
import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from chemgraph.schemas.atomsdata import AtomsData
from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema

logger = logging.getLogger(__name__)

# Reference point for CHEMGRAPH_ALLOCATION_SECONDS: the moment this module is
# imported, which is ~the moment the agent process (and thus the useful part of
# the PBS allocation) begins. CHEMGRAPH_ALLOCATION_DEADLINE (absolute epoch) is
# exact and takes precedence when both are set.
_PROCESS_START = time.time()

# Default seconds reserved before the allocation deadline for a clean stop:
# the cap trips at a step boundary, then the partial state must be flushed and
# the result file written before PBS sends SIGKILL. Overridable per run via
# CHEMGRAPH_ALLOCATION_MARGIN.
_DEFAULT_ALLOCATION_MARGIN = 60.0


def _allocation_deadline() -> Optional[float]:
    """Absolute wall-clock time (epoch seconds) when the PBS allocation ends.

    Read from the environment so a batch script can advertise the allocation's
    walltime to every calculation in the run without any per-call plumbing:

    * ``CHEMGRAPH_ALLOCATION_DEADLINE`` -- absolute epoch seconds of the kill.
      Exact, and the recommended form for real PBS jobs; e.g.
      ``export CHEMGRAPH_ALLOCATION_DEADLINE=$(date -d "+${WALL}sec" +%s)``.
    * ``CHEMGRAPH_ALLOCATION_SECONDS`` -- total budget in seconds, measured from
      this process's start (module import). Convenient when a script only knows a
      duration, but under-protects if a long setup (conda activation, model
      downloads) runs between the PBS walltime clock starting and this module
      being imported -- that gap is not counted, so prefer the DEADLINE form for
      production.

    Returns None when neither is set (no allocation cap; an explicit
    ``max_wall_seconds`` still applies on its own). Non-finite values (nan/inf)
    are rejected like non-numeric ones, so a garbage env var can never silently
    disable the cap.
    """
    raw_deadline = os.environ.get("CHEMGRAPH_ALLOCATION_DEADLINE")
    if raw_deadline:
        try:
            value = float(raw_deadline)
        except ValueError:
            logger.warning(
                "Ignoring non-numeric CHEMGRAPH_ALLOCATION_DEADLINE=%r", raw_deadline
            )
        else:
            if math.isfinite(value):
                # A deadline already in the past when THIS process started cannot
                # belong to the current allocation -- it is a leftover
                # CHEMGRAPH_ALLOCATION_DEADLINE from a prior run still in the
                # environment. Honoring it would clamp every calc to 0.001 s and
                # cap immediately, forever (same no-progress failure class as a
                # stale restart). Ignore it and fall through to SECONDS; a
                # deadline after _PROCESS_START that is merely spent mid-run is
                # still honored (and clamps in _effective_wall_seconds).
                if value < _PROCESS_START:
                    logger.warning(
                        "Ignoring stale CHEMGRAPH_ALLOCATION_DEADLINE=%r (before "
                        "this process started; leftover from a prior allocation)",
                        raw_deadline,
                    )
                else:
                    return value
            else:
                logger.warning(
                    "Ignoring non-finite CHEMGRAPH_ALLOCATION_DEADLINE=%r",
                    raw_deadline,
                )
    raw_seconds = os.environ.get("CHEMGRAPH_ALLOCATION_SECONDS")
    if raw_seconds:
        try:
            value = float(raw_seconds)
        except ValueError:
            logger.warning(
                "Ignoring non-numeric CHEMGRAPH_ALLOCATION_SECONDS=%r", raw_seconds
            )
        else:
            if math.isfinite(value):
                return _PROCESS_START + value
            logger.warning(
                "Ignoring non-finite CHEMGRAPH_ALLOCATION_SECONDS=%r", raw_seconds
            )
    return None


def _allocation_margin() -> float:
    """Seconds to reserve before the allocation deadline for a clean stop.

    Overridable via ``CHEMGRAPH_ALLOCATION_MARGIN`` (seconds); defaults to
    ``_DEFAULT_ALLOCATION_MARGIN``. A larger margin also absorbs one in-flight
    step that is already running when the deadline check fires.
    """
    raw = os.environ.get("CHEMGRAPH_ALLOCATION_MARGIN")
    if raw:
        try:
            value = float(raw)
        except ValueError:
            logger.warning(
                "Ignoring non-numeric CHEMGRAPH_ALLOCATION_MARGIN=%r", raw
            )
        else:
            if math.isfinite(value):
                return max(0.0, value)
            logger.warning(
                "Ignoring non-finite CHEMGRAPH_ALLOCATION_MARGIN=%r", raw
            )
    return _DEFAULT_ALLOCATION_MARGIN


def _effective_wall_seconds(
    explicit: Optional[float], start_time: float
) -> Optional[float]:
    """Combine an explicit ``max_wall_seconds`` with the PBS allocation budget.

    The effective cap is the *tighter* of:

    * the user's explicit ``max_wall_seconds`` (if any), and
    * the allocation's remaining time minus a safety margin (if the allocation
      env vars are set).

    This is the enforcement side of the Layer-2 auto-continue gate: even with no
    explicit cap, a calculation inside a PBS allocation self-terminates with a
    resumable partial before walltime kills it, while a short calculation
    finishes well within the budget and is never capped. The cap is *soft* -- it
    fires only at ASE step boundaries, so the margin must exceed one step's wall
    time (the 60 s default suits fast engines; DFT should raise it).

    Returns None when neither bound is set (uncapped -- original behavior), or a
    small positive (``0.001``) when the allocation is already spent ("cap
    immediately").
    """
    bounds: list[float] = []
    if explicit and explicit > 0:
        bounds.append(float(explicit))

    deadline = _allocation_deadline()
    if deadline is not None:
        alloc_remaining = deadline - _allocation_margin() - start_time
        # Clamp to a small positive: a value <= 0 would be falsy at the gate
        # (disabling the cap -- the opposite of intended) when we are in fact out
        # of time. A tiny positive makes an already-exhausted allocation cap as
        # early as possible -- typically before step 1, in which case no restart
        # file is written and the result is a no-progress partial (see the opt/
        # vib cap sites, which advertise restart_file only when a step actually
        # persisted state).
        bounds.append(max(alloc_remaining, 0.001))

    if not bounds:
        return None
    return min(bounds)


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


# ---------------------------------------------------------------------------
# Vibrational-analysis wall-clock cap
# ---------------------------------------------------------------------------

def _run_vibrations_capped(vib, deadline: Optional[float]) -> tuple[bool, int, int]:
    """Run an ASE ``Vibrations``/``Infrared`` calculation with an optional cap.

    This mirrors ``ase.vibrations.Vibrations.run`` (iterate displacements, take a
    per-displacement exclusive-create lock, compute forces, save), but adds two
    things the stock ``run()`` lacks:

    * a wall-clock deadline checked *between* displacements, so the calculation
      stops cleanly before the scheduler SIGKILLs it mid-run;
    * durability: because the caller uses a persistent cache directory (not a
      ``TemporaryDirectory``), a capped run leaves its completed displacements on
      disk and a later re-run skips them (the lock returns ``None`` for a
      displacement whose ``cache.<name>.json`` already exists).

    The deadline is checked *before* acquiring the lock, never inside it: the lock
    exclusively creates the cache file up front, so breaking mid-lock would leave
    an empty file that ASE would later mistake for a completed displacement. A
    displacement already present in the cache is skipped for free (no deadline
    check), so a complete-but-expired cache is *not* reported as capped.

    Parameters
    ----------
    vib : ase.vibrations.Vibrations
        A ``Vibrations`` (or ``Infrared``) instance bound to a persistent cache.
    deadline : Optional[float]
        Absolute ``time.time()`` value after which no new displacement starts.
        ``None`` disables the cap (all displacements run to completion).

    Returns
    -------
    tuple[bool, int, int]
        ``(capped, n_done, n_total)`` where ``capped`` is True iff the run
        stopped early with displacements still outstanding.
    """
    from ase.parallel import world

    # An earlier interrupted run (e.g. a hard kill mid-lock) can leave empty
    # cache files; drop them so those displacements are recomputed, not trusted.
    with contextlib.suppress(Exception):
        vib.cache.strip_empties()

    n_total = vib.ndof * vib.nfree + 1  # 6N+1 for nfree=2
    n_done = 0
    capped = False
    for disp, disp_atoms in vib.iterdisplace(inplace=False):
        if disp.name in vib.cache:
            # Already computed on a previous allocation -> free skip.
            n_done += 1
            continue
        if deadline is not None and time.time() >= deadline:
            capped = True
            break
        with vib.cache.lock(disp.name) as handle:
            if handle is None:
                # Won by another worker between the membership check and the
                # lock; count it as done.
                n_done += 1
                continue
            result = vib.calculate(disp_atoms, disp)
            if world.rank == 0:
                handle.save(result)
            n_done += 1
    return capped, n_done, n_total


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

    # Effective wall-clock cap: the tighter of the user's explicit
    # max_wall_seconds and the remaining PBS allocation budget (minus a safety
    # margin), so a calculation self-terminates with a resumable partial before
    # walltime kills it -- even when no explicit cap was requested. None =
    # uncapped (neither an explicit cap nor an allocation deadline is set).
    effective_wall_seconds = _effective_wall_seconds(params.max_wall_seconds, start_time)
    if (
        effective_wall_seconds is not None
        and effective_wall_seconds != params.max_wall_seconds
    ):
        logger.info(
            "Effective wall-clock cap: %.3fs (explicit max_wall_seconds=%s, "
            "allocation-bounded).",
            effective_wall_seconds,
            params.max_wall_seconds,
        )

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
        energy = atoms.get_potential_energy()
        logger.info("Single-point energy: %s eV", energy)
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
        logger.info("Results saved to %s (wall_time=%.2fs)", output_results_file, wall_time)

        if driver == "energy":
            return {
                "status": "success",
                "message": f"Simulation completed. Results saved to {os.path.abspath(output_results_file)}",
                "single_point_energy": energy,
                "unit": "eV",
                "result_file": os.path.abspath(output_results_file),
                "wall_time": wall_time,
                "wall_time_capped": False,
            }
        else:  # dipole
            return {
                "status": "success",
                "message": f"Simulation completed. Results saved to {os.path.abspath(output_results_file)}",
                "dipole_moment": dipole,
                "dipole_unit": "e * Angstrom",
                "result_file": os.path.abspath(output_results_file),
                "wall_time": wall_time,
                "wall_time_capped": False,
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
        mol_stem = Path(input_structure_file).stem if input_structure_file else "mol"
        opt_capped = False
        restart_path = None
        resume_input_file = None
        if len(atoms) > 1:
            # Soft wall-clock cap: when an effective cap is in force (an explicit
            # max_wall_seconds and/or a PBS allocation budget), step the optimizer
            # via irun() and stop cleanly at the deadline, leaving a resumable
            # partial (BFGS dumps its Hessian to restart_path and the trajectory
            # each step). When neither is set, the path below is byte-for-byte the
            # original dyn.run() behavior.
            if effective_wall_seconds:
                deadline = start_time + effective_wall_seconds
                restart_target = _resolve_path(f"{mol_stem}_opt.restart.json")
                traj_path = _resolve_path(f"{mol_stem}_opt.traj")
                dyn = optimizer_class(
                    atoms, restart=restart_target, trajectory=traj_path
                )
                converged = False
                # irun() yields is_converged at each point, and BFGS dumps its
                # restart state inside each completed step. When a step completes
                # within the budget, the capped run has a restart file on disk and
                # we advertise it. irun() also yields once before step 1, so a cap
                # that fires there leaves 0 steps and no partial to resume; that
                # case advertises no restart_file, handled by the n_done check below.
                for converged in dyn.irun(fmax=fmax, steps=steps):
                    # Convergence wins over the deadline: a step that reports a
                    # converged geometry is finished, so break without capping
                    # even when the deadline has already passed.
                    if converged:
                        break
                    if time.time() >= deadline:
                        opt_capped = True
                        break
                if opt_capped:
                    n_done = dyn.get_number_of_steps()
                    # Only claim a restart file if a step actually wrote one.
                    if n_done > 0 and os.path.isfile(restart_target):
                        restart_path = restart_target
                        # Persist the moved geometry as a standalone, ASE-readable
                        # structure so a resume continues from the partial instead
                        # of re-reading the original input. restart_target is the
                        # BFGS Hessian JSON (not a structure); this xyz is what a
                        # resumed run feeds back as input_structure_file.
                        from ase.io import write

                        resume_input_file = _resolve_path(f"{mol_stem}_opt.partial.xyz")
                        write(resume_input_file, atoms)
                    logger.warning(
                        "Optimization capped at effective wall-clock=%.3fs after "
                        "%d steps (not converged); partial geometry%s saved.",
                        effective_wall_seconds,
                        n_done,
                        " + restart" if restart_path else "",
                    )
            else:
                dyn = optimizer_class(atoms)
                converged = dyn.run(fmax=fmax, steps=steps)
        else:
            converged = True
        logger.info("Optimization converged=%s (capped=%s)", converged, opt_capped)

        single_point_energy = float(atoms.get_potential_energy())
        logger.info("Post-optimization energy: %s eV", single_point_energy)
        final_structure = AtomsData(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
        )
        thermo_data: dict = {}
        vib_data: dict = {}
        ir_data: dict = {}
        vib_capped = False
        vib_cache_path: Optional[str] = None
        ir_plot_path: Optional[str] = None

        # --------------------------------------------------------------
        # Vibrational / thermo / IR analysis
        # --------------------------------------------------------------
        # A capped optimization leaves a half-optimized geometry; running
        # vibrations on it would produce meaningless (often imaginary) modes, so
        # hand off for resume and skip the analysis on this non-stationary structure.
        if driver in {"vib", "thermo", "ir"} and opt_capped:
            logger.warning(
                "Optimization was capped before convergence; skipping %s "
                "analysis (needs a converged geometry). Resume to finish opt "
                "first.",
                driver,
            )
        elif driver in {"vib", "thermo", "ir"}:
            logger.info("Starting vibrational analysis (driver=%s)", driver)
            from ase.vibrations import Vibrations
            from ase import units

            mol_stem = (
                Path(input_structure_file).stem if input_structure_file else "mol"
            )

            # Wall-clock cap for the 6N+1 displacement evaluations. When set, use
            # a persistent, deterministic cache dir so a capped run is resumable:
            # a re-run with more budget skips already-computed displacements. When
            # unset, keep the original ephemeral TemporaryDirectory (byte-for-byte
            # the previous behavior: no cache to leave behind, no stale reuse).
            deadline = (
                start_time + effective_wall_seconds
                if effective_wall_seconds
                else None
            )
            if deadline is not None:
                vib_cache_dir = _resolve_path(f"{mol_stem}_vibcache")
                os.makedirs(vib_cache_dir, exist_ok=True)
                cache_ctx: object = contextlib.nullcontext(vib_cache_dir)
            else:
                cache_ctx = tempfile.TemporaryDirectory(
                    prefix=f"chemgraph_vib_{mol_stem}_"
                )

            with cache_ctx as tmpdir:
                vib_name = os.path.join(tmpdir, "vib")
                vib = Vibrations(atoms, name=vib_name)
                if deadline is None:
                    # Fresh cache each uncapped run (matches original behavior).
                    vib.clean()
                vib_capped, n_done, n_total = _run_vibrations_capped(vib, deadline)

                if vib_capped:
                    # Partial displacements are on disk in the persistent cache;
                    # frequencies cannot be computed until all are done, so hand
                    # off. Point restart_file at the cache dir for resume.
                    vib_cache_path = os.path.abspath(tmpdir)
                    logger.warning(
                        "Vibrational analysis CAPPED at effective wall-clock=%.3fs "
                        "after %d/%d displacements; cache kept at %s for resume.",
                        effective_wall_seconds,
                        n_done,
                        n_total,
                        vib_cache_path,
                    )
                else:
                    logger.info("Vibrational analysis complete")

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
                        if deadline is None:
                            ir.clean()
                        ir_capped, _ir_done, _ir_total = _run_vibrations_capped(
                            ir, deadline
                        )
                        if ir_capped:
                            vib_capped = True
                            vib_cache_path = os.path.abspath(tmpdir)
                            logger.warning(
                                "IR analysis CAPPED at effective wall-clock=%.3fs; "
                                "cache kept at %s for resume.",
                                effective_wall_seconds,
                                vib_cache_path,
                            )

                    if driver == "ir" and not vib_capped:
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
                        ir_data["IR Plot"] = (
                            f"Saved to {os.path.abspath(ir_plot_path)}"
                        )
                        ir_data["Normal mode data"] = (
                            f"Normal modes saved as individual .traj files with prefix {mol_stem}_"
                        )

                # ---- Thermochemistry ----
                if driver == "thermo" and not vib_capped:
                    logger.info("Computing thermochemistry (T=%s K, P=%s Pa)", temperature, pressure)
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

                        # IdealGasThermo expects total spin S; calculators expose
                        # multiplicity (2S+1) via get_multiplicity() when supported.
                        multiplicity = (
                            getattr(calc_model, "get_multiplicity", lambda: None)()
                            or 1
                        )
                        spin_S = (multiplicity - 1) / 2.0

                        thermo = IdealGasThermo(
                            vib_energies=energies,
                            potentialenergy=single_point_energy,
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
            simulation_input=params,
            vibrational_frequencies=vib_data,
            thermochemistry=thermo_data,
            success=True,
            ir_data=ir_data,
            single_point_energy=single_point_energy,
            wall_time=wall_time,
            wall_time_capped=opt_capped or vib_capped,
            restart_file=(
                restart_path if opt_capped else vib_cache_path if vib_capped else None
            ),
        )
        with open(output_results_file, "w", encoding="utf-8") as wf:
            wf.write(simulation_output.model_dump_json(indent=4))

        # ---- minimal return payload ----
        abs_output = os.path.abspath(output_results_file)
        if driver == "opt":
            if opt_capped:
                if restart_path:
                    msg = (
                        "Optimization CAPPED at the wall-clock limit (not "
                        f"converged). Partial geometry saved to {resume_input_file}. "
                        "Resume by re-running opt with input_structure_file set to "
                        "that partial geometry."
                    )
                else:
                    # Capped before completing step 1 (e.g. an already-spent
                    # allocation): no restart state was written, so there is no
                    # partial to resume from -- rerun with more budget.
                    msg = (
                        "Optimization CAPPED at the wall-clock limit before any "
                        "step completed; no restart file was written. Rerun with "
                        f"more wall-clock budget. Results saved to {abs_output}."
                    )
            else:
                msg = f"Simulation completed. Results saved to {abs_output}"
            return {
                "status": "success",
                "message": msg,
                "single_point_energy": single_point_energy,
                "unit": "eV",
                "wall_time_capped": opt_capped,
                "result_file": abs_output,
                "wall_time": wall_time,
                "restart_file": restart_path,
                "resume_input_file": resume_input_file,
            }

        # vib/thermo/ir that could not finish within the wall-clock budget (the
        # optimization capped, or the displacement sweep did): no frequencies to
        # report, but progress is durable. Re-running the same driver on the same
        # structure with more budget resumes from the saved cache.
        if driver in {"vib", "thermo", "ir"} and (opt_capped or vib_capped):
            if opt_capped:
                reason = (
                    "the pre-analysis optimization hit the wall-clock limit "
                    "before converging"
                )
            else:
                reason = "the displacement sweep hit the wall-clock limit"
            return {
                "status": "success",
                "message": (
                    f"{driver} analysis CAPPED: {reason}. No frequencies yet; "
                    f"partial progress saved to {abs_output}. Resume the same "
                    f"'{driver}' run with more wall-clock budget (a larger "
                    "max_wall_seconds, or a fresh allocation) to continue."
                ),
                "wall_time_capped": True,
                "result_file": abs_output,
                "wall_time": wall_time,
                # vib/thermo/ir resume from the durable cache dir by re-running the
                # same driver on the same input; restart_file carries the cache dir
                # (opt cap, if that is what tripped, carries its Hessian JSON). No
                # separate partial-geometry file for the analysis sweep.
                "restart_file": restart_path if opt_capped else vib_cache_path,
                "resume_input_file": resume_input_file,
            }

        if driver == "vib":
            return {
                "status": "success",
                "result": {"vibrational_frequencies": vib_data},
                "message": (
                    "Vibrational analysis completed; frequencies returned. "
                    f"Full results (structure, vibrations and metadata) saved to {abs_output}."
                ),
                "result_file": abs_output,
                "wall_time": wall_time,
                "wall_time_capped": False,
            }
        elif driver == "thermo":
            return {
                "status": "success",
                "result": {"thermochemistry": thermo_data},
                "message": (
                    "Thermochemistry computed and returned. "
                    f"Full results (structure, vibrations, thermochemistry and metadata) saved to {abs_output}"
                ),
                "result_file": abs_output,
                "wall_time": wall_time,
                "wall_time_capped": False,
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
                "result_file": abs_output,
                "wall_time": wall_time,
                "wall_time_capped": False,
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
