"""Prepare, execute, and parse isolated H2O gRASPA-SYCL simulations."""

from __future__ import annotations

from importlib.resources import files
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import warnings

import numpy as np
from ase.io import read as ase_read

from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.utils.executables import resolve_executable

logger = logging.getLogger(__name__)
TEMPLATE_FILES = (
    "simulation.input",
    "H2O.def",
    "force_field.def",
    "force_field_mixing_rules.def",
    "pseudo_atoms.def",
)


def _failure(result: dict, exc: Exception) -> dict:
    result.update(
        status="failure",
        uptake_in_mol_kg=None,
        error_type=type(exc).__name__,
        message=str(exc),
    )
    logger.warning("gRASPA failed: %s", exc)
    return result


def _read_graspa_sycl_output(
    output_path: str,
    adsorbate: str = "H2O",
    cifname: str | None = None,
    output_fname: str = "raspa.log",
    temperature: float | None = None,
    pressure: float | None = None,
) -> dict:
    """Parse the SYCL UnitCells and Overall: Average sections into mol/kg.

    This standalone parser cannot establish process completion. The runner
    additionally requires a zero exit status before accepting parsed results.
    """
    directory = Path(output_path).resolve()
    candidates = (
        list(directory.glob("*.cif"))
        if cifname is None
        else [directory / f"{cifname}.cif"]
    )
    cif_path = candidates[0] if len(candidates) == 1 else None
    result = {
        "status": "failure",
        "uptake_in_mol_kg": None,
        "adsorbate": adsorbate,
        "temperature_in_K": temperature,
        "pressure_in_Pa": pressure,
        "cif_path": str(cif_path) if cif_path else None,
    }
    try:
        if cif_path is None:
            raise ValueError(f"Could not resolve a single CIF in {directory}")
        unitcell_line = uptake_line = None
        with (directory / Path(output_fname).name).open() as stream:
            for line in stream:
                if "UnitCells" in line:
                    unitcell_line = line
                elif "Overall: Average:" in line:
                    uptake_line = line
        if unitcell_line is None or uptake_line is None:
            raise ValueError("Missing UnitCells or Overall: Average output section")
        # The SYCL output prefixes the three replication factors. Take only
        # the trailing numeric triple, also accepting its echoed input form.
        cells = [float(token) for token in unitcell_line.split()[-3:]]
        if len(cells) != 3 or any(
            not math.isfinite(v) or v <= 0 or not v.is_integer() for v in cells
        ):
            raise ValueError("UnitCells must contain three positive integers")
        token = uptake_line.split("Overall: Average:", 1)[1].split()[0].rstrip(",;")
        molecules = float(token)
        if not math.isfinite(molecules) or molecules < 0:
            raise ValueError("Uptake must be finite and nonnegative")
        atoms = ase_read(cif_path)
        _calculate_cell_size(atoms)
        mass = float(sum(atoms.get_masses())) * math.prod(cells)
        if not math.isfinite(mass) or mass <= 0:
            raise ValueError("Framework mass must be finite and positive")
        uptake = molecules / mass * 1000
        if not math.isfinite(uptake):
            raise ValueError("Converted uptake is nonfinite")
        result.update(status="success", uptake_in_mol_kg=uptake)
    except Exception as exc:
        _failure(result, exc)
    return result


def mock_graspa(params: graspa_input_schema) -> dict:
    """Return clearly marked deterministic H2O test data, without running gRASPA."""
    return {
        "status": "success",
        "is_mock": True,
        "uptake_in_mol_kg": 1.0,
        "adsorbate": params.adsorbate,
        "temperature_in_K": params.temperature,
        "pressure_in_Pa": params.pressure,
        "cif_path": params.input_structure_file,
        "input_structure_file": params.input_structure_file,
    }


def _calculate_cell_size(atoms, cutoff: float = 12.8) -> list[int]:
    """Replicate each cell width to at least twice the interaction cutoff."""
    cell = np.asarray(atoms.cell)
    volume = abs(float(np.linalg.det(cell)))
    if not np.isfinite(cell).all() or not math.isfinite(volume) or volume <= 0:
        raise ValueError("CIF must have a finite, nondegenerate unit cell")
    widths = [
        volume / np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3]))
        for i in range(3)
    ]
    if any(not math.isfinite(w) or w <= 0 for w in widths):
        raise ValueError("CIF must have positive finite cell widths")
    return [int(np.ceil(2 * cutoff / w)) for w in widths]


def resolve_output_directory(params) -> Path:
    """Resolve the run root on the executing host, retaining legacy parents."""
    from chemgraph.tools.ase_core import _resolve_path

    legacy_parent = Path(params.output_result_file).parent
    if legacy_parent != Path("."):
        warnings.warn(
            "Directory-qualified output_result_file is deprecated; use output_directory. "
            "The log is placed inside a unique run directory under that parent.",
            FutureWarning,
            stacklevel=2,
        )
    root = params.output_directory or (
        str(legacy_parent) if legacy_parent != Path(".") else "graspa_runs"
    )
    return Path(_resolve_path(os.path.expanduser(root))).resolve()


def run_graspa_core(params: graspa_input_schema) -> dict:
    """Run in a fresh directory and return results or a diagnostic failure.

    Invalid input paths raise before preparation. Prepared runs always retain
    logs and an atomic results.json, including process and parsing failures.
    """
    from chemgraph.tools.ase_core import _resolve_existing_path

    params = graspa_input_schema.model_validate(params.model_dump())
    source = Path(_resolve_existing_path(params.input_structure_file)).resolve()
    if not source.is_file() or source.suffix.lower() != ".cif":
        raise ValueError(f"Input must be an existing CIF file: {source}")
    root = resolve_output_directory(params)
    root.mkdir(parents=True, exist_ok=True)
    prefix = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:60] + "-"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=root))
    # Keep a recognizable, engine-safe filename and the exact source identity.
    copied_cif = run_dir / f"{prefix[:-1]}.cif"
    stdout = run_dir / Path(params.output_result_file).name
    stderr = run_dir / "raspa.err"
    results_path = run_dir / "results.json"
    result = {
        "status": "failure",
        "uptake_in_mol_kg": None,
        "adsorbate": params.adsorbate,
        "temperature_in_K": params.temperature,
        "pressure_in_Pa": params.pressure,
        "cif_path": str(copied_cif),
        "input_structure_file": str(source),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "stdout_path": str(stdout),
        "stderr_path": str(stderr),
        "results_path": str(results_path),
        "returncode": None,
        "wall_time": None,
    }
    started = time.monotonic()
    try:
        stdout.touch()
        stderr.touch()
        executable = resolve_executable("sycl.out", "CHEMGRAPH_GRASPA_EXECUTABLE")
        shutil.copyfile(source, copied_cif)
        cells = _calculate_cell_size(ase_read(copied_cif))
        assets = files("chemgraph.tools.files.template_graspa_sycl")
        for name in TEMPLATE_FILES:
            (run_dir / name).write_bytes(assets.joinpath(name).read_bytes())
        replacements = {
            "NCYCLE": str(params.n_cycles),
            "ADSORBATE": params.adsorbate,
            "TEMPERATURE": str(params.temperature),
            "PRESSURE": str(params.pressure),
            "UC_X UC_Y UC_Z": " ".join(map(str, cells)),
            "CUTOFF": "12.8",
            "CIFFILE": copied_cif.stem,
        }
        input_path = run_dir / "simulation.input"
        template = input_path.read_text()
        for key, value in replacements.items():
            template = template.replace(key, value)
        input_path.write_text(template)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
        with stdout.open("w") as out, stderr.open("w") as err:
            process = subprocess.run(
                [executable],
                cwd=run_dir,
                stdout=out,
                stderr=err,
                env=env,
                timeout=params.timeout_seconds,
                check=False,
            )
        result["returncode"] = process.returncode
        if process.returncode != 0:
            raise RuntimeError(
                f"gRASPA exited with code {process.returncode}; see {stderr}"
            )
        result.update(
            _read_graspa_sycl_output(
                str(run_dir),
                params.adsorbate,
                copied_cif.stem,
                stdout.name,
                params.temperature,
                params.pressure,
            )
        )
    except Exception as exc:
        _failure(result, exc)
    result["wall_time"] = time.monotonic() - started
    temporary = results_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(results_path)
    return result
