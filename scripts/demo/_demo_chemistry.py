"""Shared chemistry-screening helpers for scripts/demo/*.

Each demo script in this directory is a thin wrapper around
``submit_and_collect`` -- the actual chemistry workload (a 5-molecule
thermochemistry screen) lives here so we don't duplicate it across
backends.

Workload
--------
For each of {water, methane, ammonia, CO2, ethanol} a single MACE
``driver="thermo"`` job is submitted via the configured
``ExecutionBackend``. This drives ``chemgraph.mcp.mace_mcp_hpc._mace_worker``
under the hood, which itself wraps ``chemgraph.tools.parsl_tools.run_mace_core``
-> ``chemgraph.tools.ase_core.run_ase_core``. The ``thermo`` driver
optimises the geometry, computes vibrational frequencies, then derives
ideal-gas thermochemistry at the requested temperature/pressure
(``src/chemgraph/tools/ase_core.py:556-602``).

Two modes
---------
* ``inline=False`` -- the worker reads ``input_structure_file`` from a
  shared filesystem (local, Parsl on a compute node, EL). The demo
  reads the on-disk ``output_result_file`` JSON after the future
  resolves.
* ``inline=True``  -- the structure is embedded in the payload via
  ``atoms_to_atomsdata`` (Globus Compute, where the worker has no
  access to the laptop FS). The worker materialises the structure in a
  temp dir, runs MACE, then attaches the on-disk JSON back to the
  result as ``full_output`` (see ``mace_mcp_hpc.py:127-131``). The demo
  reads from ``raw["full_output"]``.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

MOLECULE_NAMES: list[str] = ["water", "methane", "ammonia", "co2", "ethanol"]
_HERE = Path(__file__).resolve().parent
_STRUCTURES_DIR = _HERE / "structures"


def molecule_xyz_path(name: str) -> Path:
    """Absolute path to the .xyz fixture for *name*."""
    p = _STRUCTURES_DIR / f"{name}.xyz"
    if not p.is_file():
        raise FileNotFoundError(f"Missing structure fixture: {p}")
    return p


def structures_dir() -> Path:
    """Directory holding the per-molecule .xyz fixtures."""
    return _STRUCTURES_DIR


def build_thermo_job(
    name: str,
    *,
    device: str,
    output_dir: Path,
    inline: bool,
    model: str = "medium-mpa-0",
    temperature: float = 298.15,
    pressure: float = 101325.0,
    fmax: float = 0.01,
    steps: int = 200,
) -> dict:
    """Build the job dict consumed by ``_mace_worker`` for one molecule.

    For ``inline=True`` the structure is embedded and the
    ``output_result_file`` is left relative so the worker writes into
    its own temp dir (and the on-disk JSON is shipped back to the
    caller via the ``full_output`` key).
    """
    xyz = molecule_xyz_path(name)
    job: dict[str, Any] = {
        "input_structure_file": str(xyz),
        "driver": "thermo",
        "model": model,
        "device": device,
        "temperature": temperature,
        "pressure": pressure,
        "fmax": fmax,
        "steps": steps,
        "optimizer": "lbfgs",
    }
    if inline:
        # Worker resolves the (relative) output path against its own
        # tempdir -- see mace_mcp_hpc._mace_worker:117-120.
        job["output_result_file"] = f"{name}_thermo.json"
        from ase.io import read as ase_read

        from chemgraph.tools.ase_core import atoms_to_atomsdata

        atoms = ase_read(str(xyz))
        job["inline_structure"] = atoms_to_atomsdata(atoms).model_dump()
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        job["output_result_file"] = str(
            (Path(output_dir) / f"{name}_thermo.json").resolve()
        )
    return job


def _read_full_output(raw: dict, job: dict, *, inline: bool) -> dict:
    """Return the full ASEOutputSchema dict for one finished job.

    Inline jobs carry the JSON back inline via ``full_output``.
    Non-inline jobs leave it on the shared filesystem at
    ``job["output_result_file"]``.
    """
    if inline and isinstance(raw.get("full_output"), dict):
        return raw["full_output"]
    out_file = job.get("output_result_file")
    if out_file and os.path.isfile(out_file):
        with open(out_file) as fh:
            return json.load(fh)
    return {}


def _extract_properties(name: str, raw: dict, job: dict, *, inline: bool) -> dict:
    """Pull the chemistry summary fields out of one job's result."""
    full = _read_full_output(raw, job, inline=inline)
    thermo = full.get("thermochemistry") or {}
    vib = full.get("vibrational_frequencies") or {}
    return {
        "molecule": name,
        "status": raw.get("status", "?"),
        "n_atoms": len(full.get("final_structure", {}).get("numbers", []))
        if isinstance(full.get("final_structure"), dict)
        else None,
        "energy_eV": full.get("single_point_energy"),
        "enthalpy_eV": thermo.get("enthalpy"),
        "entropy_eV_per_K": thermo.get("entropy"),
        "gibbs_free_energy_eV": thermo.get("gibbs_free_energy"),
        "n_frequencies": (
            len(vib.get("frequencies", []))
            if isinstance(vib, dict) and isinstance(vib.get("frequencies"), list)
            else None
        ),
        "converged": full.get("converged"),
        "wall_time_s": full.get("wall_time"),
    }


def submit_and_collect(
    backend,
    molecule_names: list[str] | None = None,
    *,
    device: str,
    output_dir: Path | str,
    inline: bool,
    timeout: float = 6000.0,
    ppn: int = 1,
) -> list[dict]:
    """Submit one MACE thermo job per molecule, gather and summarise.

    Returns a list of per-molecule property dicts in submission order.
    Raises if any future fails -- demos should *fail loud*, not swallow.
    """
    from chemgraph.execution.base import TaskSpec
    from chemgraph.mcp.mace_mcp_hpc import _mace_worker

    names = molecule_names or MOLECULE_NAMES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        build_thermo_job(name, device=device, output_dir=output_dir, inline=inline)
        for name in names
    ]
    tasks = [
        TaskSpec(
            task_id=f"demo-thermo-{name}",
            task_type="python",
            callable=_mace_worker,
            kwargs={"job": job},
            processes_per_node=ppn,
        )
        for name, job in zip(names, jobs)
    ]
    print(
        f"\nSubmitting {len(tasks)} thermo jobs to backend={type(backend).__name__} "
        f"(device={device}, inline={inline})..."
    )
    futures = backend.submit_batch(tasks)

    results: list[dict] = []
    for name, job, fut in zip(names, jobs, futures):
        print(f"  waiting on {name}...", flush=True)
        try:
            raw = fut.result(timeout=timeout)
            if not isinstance(raw, dict):
                raise RuntimeError(f"{name}: non-dict result {type(raw).__name__}: {raw!r}")
            if raw.get("status") != "success":
                raise RuntimeError(f"{name}: backend returned status={raw.get('status')!r}: {raw}")
            results.append(_extract_properties(name, raw, job, inline=inline))
        except Exception as e:
            print(f"collecting results for job {name} failed with error: {e}")
            results.append(None)
    return results


def write_csv(results: list[dict], csv_path: Path | str) -> Path:
    """Write the property table to *csv_path*. Returns the path."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        csv_path.write_text("")
        return csv_path
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    return csv_path


def print_summary(results: list[dict], title: str = "") -> None:
    """Print a fixed-width table of the screening results."""
    if title:
        print(f"\n=== {title} ===")
    if not results:
        print("(no results)")
        return
    header = (
        f"{'molecule':<10}  {'energy/eV':>12}  {'enthalpy/eV':>13}  "
        f"{'S/(eV/K)':>12}  {'G/eV':>12}  {'#freqs':>7}  {'wall/s':>8}  {'conv':>5}"
    )
    print(header)
    print("-" * len(header))

    def fmt(val, w, p=4):
        if val is None:
            return f"{'-':>{w}}"
        if isinstance(val, float):
            return f"{val:>{w}.{p}f}"
        return f"{val!s:>{w}}"

    for r in results:
        print(
            f"{r['molecule']:<10}  "
            f"{fmt(r.get('energy_eV'), 12)}  "
            f"{fmt(r.get('enthalpy_eV'), 13)}  "
            f"{fmt(r.get('entropy_eV_per_K'), 12, 6)}  "
            f"{fmt(r.get('gibbs_free_energy_eV'), 12)}  "
            f"{fmt(r.get('n_frequencies'), 7, 0)}  "
            f"{fmt(r.get('wall_time_s'), 8, 1)}  "
            f"{fmt(r.get('converged'), 5)}"
        )
    print()


def agent_prompt(device: str = "cpu") -> str:
    """Standard natural-language prompt used by all *_agent.py demos.

    The structure paths reference the demo's own ``structures/`` so the
    agent can call ``run_mace_single`` directly without staging.
    Replace the file paths if you adapt this for a different layout.
    """
    files = ", ".join(str(molecule_xyz_path(n)) for n in MOLECULE_NAMES)
    return (
        f"Using the MACE tool with driver='thermo', model='medium-mpa-0', "
        f"device='{device}', temperature=298.15 K, pressure=101325 Pa, "
        f"compute thermochemistry for each of these five molecules:\n"
        f"  - water:    {molecule_xyz_path('water')}\n"
        f"  - methane:  {molecule_xyz_path('methane')}\n"
        f"  - ammonia:  {molecule_xyz_path('ammonia')}\n"
        f"  - CO2:      {molecule_xyz_path('co2')}\n"
        f"  - ethanol:  {molecule_xyz_path('ethanol')}\n"
        f"Call run_mace_single once per molecule (do not batch them yourself). "
        f"For each result, retrieve the optimized electronic energy, enthalpy, "
        f"entropy and Gibbs free energy by reading the output JSON via "
        f"extract_output_json. After all five complete, report a markdown table "
        f"with columns: molecule, energy (eV), H (eV), G (eV), and wall-time then a one-line "
        f"observation about which molecule has the lowest Gibbs free energy.\n\n"
        f"(Structure paths for reference: {files})"
    )
