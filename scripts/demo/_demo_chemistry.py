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
import sys
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
        # Relative name only. The worker-side wrapper
        # (_make_worker_with_full_output) resolves this into a tmpdir created
        # ON the worker, so we never bake a laptop path into a remote job.
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


def _calculator_dict(calc: str | dict | None, device: str = "cpu") -> dict:
    """Normalise a ``--calculator`` value into an ASE calculator dict.

    Accepts a short name ('mace_mp', 'emt', 'tblite'/'xtb') or a ready
    calculator dict (passed through). ``None`` defaults to MACE-MP so the
    ``ase`` workload matches the ``thermo`` workload out of the box.
    """
    if isinstance(calc, dict):
        return calc
    name = (calc or "mace_mp").lower()
    if name in ("mace", "mace_mp"):
        return {"calculator_type": "mace_mp", "model": "medium-mpa-0", "device": device}
    if name == "emt":
        return {"calculator_type": "emt"}
    if name in ("tblite", "xtb", "gfn2-xtb", "gfn1-xtb"):
        return {"calculator_type": "tblite"}
    # Fall through: assume the string is a valid calculator_type.
    return {"calculator_type": name}


def build_ase_job(
    name: str,
    *,
    output_dir: Path,
    inline: bool,
    calculator: str | dict | None = None,
    driver: str = "thermo",
    device: str = "cpu",
    temperature: float = 298.15,
    pressure: float = 101325.0,
    fmax: float = 0.01,
    steps: int = 200,
    optimizer: str = "lbfgs",
) -> dict:
    """Build the job dict consumed by ``_ase_worker`` for one molecule.

    Mirrors :func:`build_thermo_job` but targets the general ASE server:
    the calculator is selectable and the output key is ``output_results_file``
    (with an 's'), matching :class:`ASEInputSchema`.
    """
    xyz = molecule_xyz_path(name)
    job: dict[str, Any] = {
        "input_structure_file": str(xyz),
        "driver": driver,
        "calculator": _calculator_dict(calculator, device),
        "temperature": temperature,
        "pressure": pressure,
        "fmax": fmax,
        "steps": steps,
        "optimizer": optimizer,
    }
    if inline:
        # Relative name only -- resolved into a worker-side tmpdir (see
        # _make_worker_with_full_output). Never bake a laptop path into a
        # remote job.
        job["output_results_file"] = f"{name}_ase.json"
        from ase.io import read as ase_read

        from chemgraph.tools.ase_core import atoms_to_atomsdata

        atoms = ase_read(str(xyz))
        job["inline_structure"] = atoms_to_atomsdata(atoms).model_dump()
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        job["output_results_file"] = str(
            (Path(output_dir) / f"{name}_ase.json").resolve()
        )
    return job


def build_fairchem_job(
    name: str,
    *,
    device: str,
    output_dir: Path,
    inline: bool,
    model_name: str = "uma-s-1p1",
    task_name: str | None = "omol",
    driver: str = "thermo",
    temperature: float = 298.15,
    pressure: float = 101325.0,
    fmax: float = 0.01,
    steps: int = 200,
    optimizer: str = "lbfgs",
) -> dict:
    """Build the job dict consumed by ``_fairchem_worker`` for one molecule.

    Mirrors :func:`build_thermo_job` but targets the FairChem/UMA server.
    Job keys match :class:`fairchem_input_schema`; the output key is
    ``output_result_file`` (as in the FairChem server). For ``inline=True``
    the structure is embedded and ``_workload="fairchem"`` is set so the
    inline read-back wrapper picks the FairChem worker.
    """
    xyz = molecule_xyz_path(name)
    job: dict[str, Any] = {
        "input_structure_file": str(xyz),
        "driver": driver,
        "model_name": model_name,
        "task_name": task_name,
        "device": device,
        "temperature": temperature,
        "pressure": pressure,
        "fmax": fmax,
        "steps": steps,
        "optimizer": optimizer,
    }
    if inline:
        # Relative name only -- resolved into a worker-side tmpdir by
        # _fairchem_worker. Never bake a laptop path into a remote job.
        job["output_result_file"] = f"{name}_fairchem.json"
        job["_workload"] = "fairchem"  # consumed by the inline wrapper
        from ase.io import read as ase_read

        from chemgraph.tools.ase_core import atoms_to_atomsdata

        atoms = ase_read(str(xyz))
        job["inline_structure"] = atoms_to_atomsdata(atoms).model_dump()
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        job["output_result_file"] = str(
            (Path(output_dir) / f"{name}_fairchem.json").resolve()
        )
    return job


def build_graspa_job(
    cif_path: str | Path,
    *,
    output_dir: Path,
    adsorbate: str = "H2O",
    temperature: float = 298.15,
    pressure: float = 101325.0,
    n_cycles: int = 10000,
) -> dict:
    """Build the job dict consumed by ``_graspa_worker`` for one CIF.

    gRASPA is HPC-only (the SYCL binary path is baked into
    ``chemgraph.tools.graspa_core``) and has no inline transport -- the
    worker reads ``input_structure_file`` from a shared/remote filesystem.
    """
    cif_path = Path(cif_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return {
        "_structure_name": cif_path.stem,
        "input_structure_file": str(cif_path),
        "output_result_file": str(
            (Path(output_dir) / f"{cif_path.stem}_raspa.log").resolve()
        ),
        "adsorbate": adsorbate,
        "temperature": temperature,
        "pressure": pressure,
        "n_cycles": n_cycles,
    }


def build_pacmof2_job(
    cif_path: str | Path,
    *,
    output_dir: Path,
    identifier: str = "_pacmof",
    adjust_charge_method: str = "mean",
    net_charge: int | float | dict = 0,
) -> dict:
    """Build the job dict consumed by ``_pacmof2_worker`` for one CIF.

    PACMOF2 mirrors gRASPA: CIF-in, shared/remote filesystem only (no
    inline transport). It writes the charged CIF next to its input, so
    ``output_dir`` is only used to keep the harness signature uniform.
    """
    cif_path = Path(cif_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return {
        "_structure_name": cif_path.stem,
        "input_structure_file": str(cif_path),
        "identifier": identifier,
        "adjust_charge_method": adjust_charge_method,
        "net_charge": net_charge,
    }


def _read_full_output(raw: dict, job: dict, *, inline: bool) -> dict:
    """Return the full ASEOutputSchema dict for one finished job.

    Inline jobs carry the JSON back inline via ``full_output``.
    Non-inline jobs leave it on the shared filesystem at
    ``job["output_result_file"]``.
    """
    if inline and isinstance(raw.get("full_output"), dict):
        return raw["full_output"]
    out_file = job.get("output_result_file") or job.get("output_results_file")
    if out_file and os.path.isfile(out_file):
        with open(out_file) as fh:
            return json.load(fh)
    return {}


def _extract_properties(name: str, raw: dict, job: dict, *, inline: bool) -> dict:
    """Pull the chemistry summary fields out of one MACE/ASE job's result.

    Both the MACE and general-ASE workloads produce an ``ASEOutputSchema``
    JSON, so this extractor serves both.
    """
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


def _extract_graspa_properties(name: str, raw: dict, job: dict, *, inline: bool) -> dict:
    """Pull the GCMC summary fields out of one gRASPA job's result.

    gRASPA writes a ``raspa.log`` *text* file, not JSON -- but the
    ``_graspa_worker`` already returns a parsed dict (uptake, T, P), so we
    read directly from ``raw`` and never try to ``json.load`` the log.
    """
    return {
        "structure": raw.get("structure") or name,
        "status": raw.get("status", "?"),
        "adsorbate": job.get("adsorbate"),
        "temperature_K": raw.get("temperature") or job.get("temperature"),
        "pressure_Pa": raw.get("pressure") or job.get("pressure"),
        "uptake_mol_per_kg": raw.get("uptake_in_mol_kg"),
        "wall_time_s": raw.get("wall_time"),
    }


def _extract_pacmof2_properties(name: str, raw: dict, job: dict, *, inline: bool) -> dict:
    """Pull the charge summary out of one PACMOF2 job's result.

    ``_pacmof2_worker`` already returns the parsed summary dict (output
    CIF path, per-element mean charges, sum of charges), so we read
    directly from ``raw`` -- no output-file round-trip (mirrors gRASPA).
    """
    return {
        "structure": raw.get("structure") or name,
        "status": raw.get("status", "?"),
        "n_atoms": raw.get("n_atoms"),
        "sum_of_charges": raw.get("sum_of_charges"),
        "charge_range": raw.get("charge_range"),
        "per_element_mean_charge": raw.get("per_element_mean_charge"),
        "output_cif_path": raw.get("output_cif_path"),
    }


def _make_worker_with_full_output():
    """Return a backend-side wrapper that runs the per-workload worker, then
    reads its output JSON back in-process and attaches it as ``full_output``.

    The wrapper handles both the ``thermo`` (MACE) and ``ase`` workloads --
    it branches on ``job["_workload"]`` and picks the matching worker and
    output-file key (MACE uses ``output_result_file``; ASE uses
    ``output_results_file``). gRASPA never takes this path (it is HPC-only
    with no inline transport).

    For inline transports (Globus Compute) the worker writes its results to a
    path on the *remote* worker that the caller cannot read. Running the
    read-back here -- on the same worker, right after the write -- ships the
    full ASEOutputSchema back inline without any src/ changes.

    The output tmpdir is chosen *inside* the wrapper (i.e. on the worker), not
    on the submitting host: otherwise a laptop temp path (e.g. macOS
    ``/var/folders/...``) would be baked into the job and fail ``os.makedirs``
    on the HPC node.

    Serialization matters here. Globus Compute (dill) pickles a function
    defined in this script *by reference* -- the remote Crux worker would try
    to ``import _demo_chemistry`` (not on its path) and fail. Two things make
    it travel *by value* instead so it runs on a worker that has never seen
    this file:

    1. ``__module__ = "__main__"`` -- dill embeds the source only for
       ``__main__`` functions.
    2. A fresh ``__globals__`` (just ``__builtins__``) via ``FunctionType`` --
       otherwise dill drags this whole module's globals (the ``_demo_chemistry``
       ModuleSpec) into the byte stream, reintroducing the by-reference import.

    All real dependencies are imported *inside* the wrapper, so only installed
    ``chemgraph`` symbols are resolved remotely and the clean globals suffice.
    """
    import types

    def _worker_with_full_output(job: dict) -> dict:
        import os
        import tempfile

        job = dict(job)  # don't mutate the shipped dict
        workload = job.pop("_workload", "thermo")
        if workload == "ase":
            from chemgraph.mcp.ase_mcp_hpc import _ase_worker as _worker
            from chemgraph.tools.ase_core import extract_output_json_core as _extract
            out_key = "output_results_file"
        elif workload == "fairchem":
            from chemgraph.mcp.fairchem_mcp_hpc import _fairchem_worker as _worker
            from chemgraph.tools.fairchem_tools import extract_output_json as _extract
            out_key = "output_result_file"
        else:
            from chemgraph.mcp.mace_mcp_hpc import _mace_worker as _worker
            from chemgraph.tools.parsl_tools import extract_output_json as _extract
            out_key = "output_result_file"

        out_file = job.get(out_key, "")
        # Inline job (relative path, no pre-staged remote file): pick a
        # concrete writable path ON THIS worker so we know where to read the
        # JSON back from. The worker preserves absolute paths verbatim.
        if (
            out_file
            and not os.path.isabs(out_file)
            and "remote_structure_file" not in job
        ):
            tmpdir = tempfile.mkdtemp(prefix="cg_demo_")
            out_file = os.path.join(tmpdir, os.path.basename(out_file))
            job[out_key] = out_file

        result = _worker(job)
        if isinstance(result, dict) and out_file:
            full = _extract(out_file)  # {} if missing/unreadable
            if full:
                result["full_output"] = full
        return result

    worker = types.FunctionType(
        _worker_with_full_output.__code__,
        {"__builtins__": __builtins__},
        "_worker_with_full_output",
    )
    worker.__module__ = "__main__"
    worker.__qualname__ = "_worker_with_full_output"
    return worker


def _thermo_package_worker():
    from chemgraph.mcp.mace_mcp_hpc import _mace_worker

    return _mace_worker


def _ase_package_worker():
    from chemgraph.mcp.ase_mcp_hpc import _ase_worker

    return _ase_worker


def _graspa_package_worker():
    from chemgraph.mcp.graspa_mcp_hpc import _graspa_worker

    return _graspa_worker


def _fairchem_package_worker():
    from chemgraph.mcp.fairchem_mcp_hpc import _fairchem_worker

    return _fairchem_worker


def _pacmof2_package_worker():
    from chemgraph.mcp.pacmof2_mcp_hpc import _pacmof2_worker

    return _pacmof2_worker


# Registry mapping a --workload value to how the shared harness runs it.
# ``inline_ok`` marks workloads that support Globus-Compute inline transport
# (structure embedded in the payload); gRASPA is HPC/shared-FS only.
WORKLOADS: dict[str, dict[str, Any]] = {
    "thermo": {
        "label": "MACE thermo",
        "package_worker": _thermo_package_worker,
        "extractor": _extract_properties,
        "inline_ok": True,
    },
    "ase": {
        "label": "ASE",
        "package_worker": _ase_package_worker,
        "extractor": _extract_properties,
        "inline_ok": True,
    },
    "graspa": {
        "label": "gRASPA GCMC",
        "package_worker": _graspa_package_worker,
        "extractor": _extract_graspa_properties,
        "inline_ok": False,
    },
    "fairchem": {
        "label": "FairChem/UMA",
        "package_worker": _fairchem_package_worker,
        "extractor": _extract_properties,
        "inline_ok": True,
    },
    "pacmof2": {
        "label": "PACMOF2 charges",
        "package_worker": _pacmof2_package_worker,
        "extractor": _extract_pacmof2_properties,
        "inline_ok": False,
    },
}


def _build_jobs(
    workload: str,
    items: list[str],
    *,
    device: str,
    output_dir: Path,
    inline: bool,
    calculator: str | dict | None = None,
    driver: str = "thermo",
    adsorbate: str = "H2O",
    temperature: float = 298.15,
    pressure: float = 101325.0,
    model_name: str = "uma-s-1p1",
    task_name: str | None = "omol",
    net_charge: int | float | dict = 0,
    identifier: str = "_pacmof",
    adjust_charge_method: str = "mean",
) -> list[dict]:
    """Build the per-item job dicts for *workload*."""
    if workload == "thermo":
        return [
            build_thermo_job(name, device=device, output_dir=output_dir, inline=inline)
            for name in items
        ]
    if workload == "ase":
        jobs = []
        for name in items:
            job = build_ase_job(
                name,
                output_dir=output_dir,
                inline=inline,
                calculator=calculator,
                driver=driver,
                device=device,
                temperature=temperature,
                pressure=pressure,
            )
            if inline:
                job["_workload"] = "ase"  # consumed by the inline wrapper
            jobs.append(job)
        return jobs
    if workload == "fairchem":
        return [
            build_fairchem_job(
                name,
                device=device,
                output_dir=output_dir,
                inline=inline,
                model_name=model_name,
                task_name=task_name,
                driver=driver,
                temperature=temperature,
                pressure=pressure,
            )
            for name in items
        ]
    if workload == "graspa":
        return [
            build_graspa_job(
                cif,
                output_dir=output_dir,
                adsorbate=adsorbate,
                temperature=temperature,
                pressure=pressure,
            )
            for cif in items
        ]
    if workload == "pacmof2":
        return [
            build_pacmof2_job(
                cif,
                output_dir=output_dir,
                identifier=identifier,
                adjust_charge_method=adjust_charge_method,
                net_charge=net_charge,
            )
            for cif in items
        ]
    raise ValueError(f"Unknown workload: {workload!r}")


# Workloads whose work items are CIF paths (vs. molecule fixture names).
_CIF_WORKLOADS = frozenset({"graspa", "pacmof2"})


def _item_label(workload: str, item: str) -> str:
    """Short display name for one work item."""
    return Path(item).stem if workload in _CIF_WORKLOADS else item


def submit_and_collect(
    backend,
    molecule_names: list[str] | None = None,
    *,
    device: str,
    output_dir: Path | str,
    inline: bool,
    workload: str = "thermo",
    items: list[str] | None = None,
    calculator: str | dict | None = None,
    driver: str = "thermo",
    adsorbate: str = "H2O",
    temperature: float = 298.15,
    pressure: float = 101325.0,
    model_name: str = "uma-s-1p1",
    task_name: str | None = "omol",
    net_charge: int | float | dict = 0,
    identifier: str = "_pacmof",
    adjust_charge_method: str = "mean",
    timeout: float = 6000.0,
    ppn: int = 1,
) -> list[dict]:
    """Submit one job per work item for *workload*, gather and summarise.

    ``workload="thermo"`` (default) runs the original 5-molecule MACE screen.
    ``workload="ase"`` runs the same molecules through the general ASE server
    with a selectable ``calculator``. ``workload="graspa"`` runs GCMC over the
    CIF paths in ``items`` (HPC-only). ``molecule_names`` is retained for
    backward compatibility and used as ``items`` for the thermo/ase workloads.

    Returns a list of per-item property dicts in submission order.
    """
    from chemgraph.execution.base import TaskSpec

    if workload not in WORKLOADS:
        raise ValueError(
            f"Unknown workload {workload!r}; choose from {sorted(WORKLOADS)}."
        )
    spec = WORKLOADS[workload]

    if inline and not spec["inline_ok"]:
        raise ValueError(
            f"Workload {workload!r} does not support inline transport "
            f"(it is shared-filesystem / HPC only)."
        )

    if items is None:
        items = molecule_names or MOLECULE_NAMES
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = _build_jobs(
        workload,
        items,
        device=device,
        output_dir=output_dir,
        inline=inline,
        calculator=calculator,
        driver=driver,
        adsorbate=adsorbate,
        temperature=temperature,
        pressure=pressure,
        model_name=model_name,
        task_name=task_name,
        net_charge=net_charge,
        identifier=identifier,
        adjust_charge_method=adjust_charge_method,
    )

    # Inline transport (Globus Compute): the worker writes results to a path
    # on the remote host the caller can't read, so use a wrapper that ships
    # the output JSON back inline as ``full_output``. Shared-FS backends
    # (local/parsl/EL) read the on-disk absolute path directly and use the
    # package-level worker, which pickles under stdlib pickle too.
    worker = _make_worker_with_full_output() if inline else spec["package_worker"]()
    labels = [_item_label(workload, it) for it in items]
    tasks = [
        TaskSpec(
            task_id=f"demo-{workload}-{label}",
            task_type="python",
            callable=worker,
            kwargs={"job": job},
            processes_per_node=ppn,
        )
        for label, job in zip(labels, jobs)
    ]
    print(
        f"\nSubmitting {len(tasks)} {spec['label']} jobs to "
        f"backend={type(backend).__name__} (device={device}, inline={inline})..."
    )
    futures = backend.submit_batch(tasks)

    extractor = spec["extractor"]
    results: list[dict] = []
    for label, job, fut in zip(labels, jobs, futures):
        print(f"  waiting on {label}...", flush=True)
        try:
            raw = fut.result(timeout=timeout)
            if not isinstance(raw, dict):
                raise RuntimeError(f"{label}: non-dict result {type(raw).__name__}: {raw!r}")
            if raw.get("status") != "success":
                raise RuntimeError(f"{label}: backend returned status={raw.get('status')!r}: {raw}")
            results.append(extractor(label, raw, job, inline=inline))
        except Exception as e:
            print(f"collecting results for job {label} failed with error: {e}")
            results.append(None)
    return results


def write_csv(results: list[dict], csv_path: Path | str) -> Path:
    """Write the property table to *csv_path*. Returns the path.

    Failed jobs appear as ``None`` in *results*; they are skipped so a
    batch where every job failed still writes a valid (empty) CSV instead
    of crashing.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r for r in results if r]
    if not rows:
        csv_path.write_text("")
        return csv_path
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _fmt(val, w, p=4):
    if val is None:
        return f"{'-':>{w}}"
    if isinstance(val, float):
        return f"{val:>{w}.{p}f}"
    return f"{val!s:>{w}}"


def _print_thermo_summary(results: list[dict]) -> None:
    header = (
        f"{'molecule':<10}  {'energy/eV':>12}  {'enthalpy/eV':>13}  "
        f"{'S/(eV/K)':>12}  {'G/eV':>12}  {'#freqs':>7}  {'wall/s':>8}  {'conv':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if not r:
            continue
        print(
            f"{r.get('molecule', '?'):<10}  "
            f"{_fmt(r.get('energy_eV'), 12)}  "
            f"{_fmt(r.get('enthalpy_eV'), 13)}  "
            f"{_fmt(r.get('entropy_eV_per_K'), 12, 6)}  "
            f"{_fmt(r.get('gibbs_free_energy_eV'), 12)}  "
            f"{_fmt(r.get('n_frequencies'), 7, 0)}  "
            f"{_fmt(r.get('wall_time_s'), 8, 1)}  "
            f"{_fmt(r.get('converged'), 5)}"
        )
    print()


def _print_graspa_summary(results: list[dict]) -> None:
    header = (
        f"{'structure':<16}  {'adsorbate':>10}  {'T/K':>8}  {'P/Pa':>12}  "
        f"{'uptake/(mol/kg)':>16}  {'wall/s':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if not r:
            continue
        print(
            f"{r.get('structure', '?'):<16}  "
            f"{_fmt(r.get('adsorbate'), 10)}  "
            f"{_fmt(r.get('temperature_K'), 8, 2)}  "
            f"{_fmt(r.get('pressure_Pa'), 12, 1)}  "
            f"{_fmt(r.get('uptake_mol_per_kg'), 16, 4)}  "
            f"{_fmt(r.get('wall_time_s'), 8, 1)}"
        )
    print()


def _print_pacmof2_summary(results: list[dict]) -> None:
    header = (
        f"{'structure':<16}  {'status':>8}  {'#atoms':>7}  {'sum(q)':>10}  "
        f"{'q range':>18}  {'output CIF':<40}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        if not r:
            continue
        crange = r.get("charge_range")
        crange_str = (
            f"[{crange[0]:.3f}, {crange[1]:.3f}]"
            if isinstance(crange, (list, tuple)) and len(crange) == 2
            else "-"
        )
        cif = r.get("output_cif_path") or "-"
        print(
            f"{r.get('structure', '?'):<16}  "
            f"{_fmt(r.get('status'), 8)}  "
            f"{_fmt(r.get('n_atoms'), 7, 0)}  "
            f"{_fmt(r.get('sum_of_charges'), 10, 4)}  "
            f"{crange_str:>18}  "
            f"{cif:<40}"
        )
    print()


def print_summary(results: list[dict], title: str = "", workload: str = "thermo") -> None:
    """Print a fixed-width table of the screening results.

    ``thermo``, ``ase`` and ``fairchem`` share the molecular-property
    table; ``graspa`` uses a GCMC uptake table; ``pacmof2`` uses a charge
    summary table.
    """
    if title:
        print(f"\n=== {title} ===")
    if not results:
        print("(no results)")
        return
    if workload == "graspa":
        _print_graspa_summary(results)
    elif workload == "pacmof2":
        _print_pacmof2_summary(results)
    else:
        _print_thermo_summary(results)


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


def agent_prompt_ase(device: str = "cpu", calculator: str = "mace_mp") -> str:
    """Prompt for the general-ASE (``run_ase_single``) agent demos."""
    files = ", ".join(str(molecule_xyz_path(n)) for n in MOLECULE_NAMES)
    return (
        f"Using the ASE tool (run_ase_single) with driver='thermo', "
        f"calculator='{calculator}', device='{device}', temperature=298.15 K, "
        f"pressure=101325 Pa, compute thermochemistry for each of these five "
        f"molecules:\n"
        f"  - water:    {molecule_xyz_path('water')}\n"
        f"  - methane:  {molecule_xyz_path('methane')}\n"
        f"  - ammonia:  {molecule_xyz_path('ammonia')}\n"
        f"  - CO2:      {molecule_xyz_path('co2')}\n"
        f"  - ethanol:  {molecule_xyz_path('ethanol')}\n"
        f"Call run_ase_single once per molecule (do not batch them yourself). "
        f"For each result, retrieve the optimized electronic energy, enthalpy, "
        f"entropy and Gibbs free energy by reading the output JSON via "
        f"extract_output_json. After all five complete, report a markdown table "
        f"with columns: molecule, energy (eV), H (eV), G (eV), and wall-time then "
        f"a one-line observation about which molecule has the lowest Gibbs free "
        f"energy.\n\n(Structure paths for reference: {files})"
    )


def agent_prompt_fairchem(device: str = "cpu", model_name: str = "uma-s-1p1") -> str:
    """Prompt for the FairChem/UMA (``run_fairchem_single``) agent demos."""
    files = ", ".join(str(molecule_xyz_path(n)) for n in MOLECULE_NAMES)
    return (
        f"Using the FairChem tool (run_fairchem_single) with driver='thermo', "
        f"model_name='{model_name}', task_name='omol', device='{device}', "
        f"temperature=298.15 K, pressure=101325 Pa, compute thermochemistry for "
        f"each of these five molecules:\n"
        f"  - water:    {molecule_xyz_path('water')}\n"
        f"  - methane:  {molecule_xyz_path('methane')}\n"
        f"  - ammonia:  {molecule_xyz_path('ammonia')}\n"
        f"  - CO2:      {molecule_xyz_path('co2')}\n"
        f"  - ethanol:  {molecule_xyz_path('ethanol')}\n"
        f"Call run_fairchem_single once per molecule (do not batch them yourself). "
        f"For each result, retrieve the optimized electronic energy, enthalpy, "
        f"entropy and Gibbs free energy by reading the output JSON via "
        f"extract_output_json. After all five complete, report a markdown table "
        f"with columns: molecule, energy (eV), H (eV), G (eV), and wall-time then "
        f"a one-line observation about which molecule has the lowest Gibbs free "
        f"energy.\n\n(Structure paths for reference: {files})"
    )


def agent_prompt_graspa(
    cif_paths: list[str],
    *,
    adsorbate: str = "H2O",
    temperature: float = 298.15,
    pressure: float = 101325.0,
) -> str:
    """Prompt for the gRASPA (``run_graspa_ensemble``) agent demos.

    gRASPA is HPC-only; ``cif_paths`` should point at CIFs reachable on the
    remote (shared or pre-staged) filesystem.
    """
    listing = "\n".join(f"  - {p}" for p in cif_paths)
    return (
        f"Using the gRASPA tool, run GCMC adsorption for adsorbate="
        f"'{adsorbate}' at temperature={temperature} K and pressure="
        f"{pressure} Pa on the following CIF structures:\n"
        f"{listing}\n"
        f"After the runs complete, report a markdown table with columns: "
        f"structure, uptake (mol/kg). Report tool errors exactly as they occur."
    )


def agent_prompt_pacmof2(cif_paths: list[str], *, net_charge: int | float = 0) -> str:
    """Prompt for the PACMOF2 (``run_pacmof2_ensemble``) agent demos.

    PACMOF2 is shared-FS/HPC-only; ``cif_paths`` should point at CIFs
    reachable on the remote (shared or pre-staged) filesystem.
    """
    listing = "\n".join(f"  - {p}" for p in cif_paths)
    return (
        f"Using the PACMOF2 tool (run_pacmof2_ensemble) with net_charge="
        f"{net_charge}, assign machine-learned partial atomic charges to the "
        f"following MOF CIF structures:\n"
        f"{listing}\n"
        f"After the runs complete, report a markdown table with columns: "
        f"structure, sum of charges, output CIF path. Report tool errors exactly "
        f"as they occur."
    )


def prompt_for(workload: str, *, device: str = "cpu", calculator: str = "mace_mp") -> str:
    """Return the agent prompt for *workload*.

    ``thermo``/``ase``/``fairchem`` build molecule prompts here; ``graspa``
    and ``pacmof2`` need explicit CIF paths (call their dedicated builders).
    """
    if workload == "ase":
        return agent_prompt_ase(device=device, calculator=calculator)
    if workload == "fairchem":
        return agent_prompt_fairchem(device=device)
    if workload == "graspa":
        raise ValueError(
            "gRASPA prompts need explicit CIF paths; call agent_prompt_graspa()."
        )
    if workload == "pacmof2":
        raise ValueError(
            "PACMOF2 prompts need explicit CIF paths; call agent_prompt_pacmof2()."
        )
    return agent_prompt(device=device)


# ── Shared CLI helpers for the demo_*_direct.py / demo_*_agent.py wrappers ──

# gRASPA needs the SYCL binary baked into chemgraph.tools.graspa_core, so it
# only runs on HPC-capable backends.
GRASPA_HPC_BACKENDS = frozenset({"parsl", "ensemble_launcher", "globus_compute"})

# Which MCP server module + default port + single-call tool each workload uses.
# The agent demos spawn these over stdio (the port is informational for stdio).
MCP_SERVER_BY_WORKLOAD: dict[str, dict[str, Any]] = {
    "thermo": {
        "module": "chemgraph.mcp.mace_mcp_hpc",
        "port": 9004,
        "label": "ChemGraph MACE",
        "single_tool": "run_mace_single",
    },
    "ase": {
        "module": "chemgraph.mcp.ase_mcp_hpc",
        "port": 9005,
        "label": "ChemGraph ASE",
        "single_tool": "run_ase_single",
    },
    "graspa": {
        "module": "chemgraph.mcp.graspa_mcp_hpc",
        "port": 9001,
        "label": "ChemGraph gRASPA",
        "single_tool": "run_graspa_ensemble",
    },
    "fairchem": {
        "module": "chemgraph.mcp.fairchem_mcp_hpc",
        "port": 9008,
        "label": "ChemGraph FairChem",
        "single_tool": "run_fairchem_single",
    },
    "pacmof2": {
        "module": "chemgraph.mcp.pacmof2_mcp_hpc",
        "port": 9009,
        "label": "ChemGraph PACMOF2",
        "single_tool": "run_pacmof2_ensemble",
    },
}


def mcp_server_for(workload: str) -> dict[str, Any]:
    """Return the MCP server spec (module/port/label/tool) for *workload*."""
    try:
        return MCP_SERVER_BY_WORKLOAD[workload]
    except KeyError:
        raise ValueError(
            f"Unknown workload {workload!r}; choose from "
            f"{sorted(MCP_SERVER_BY_WORKLOAD)}."
        ) from None


def add_workload_args(parser) -> None:
    """Add the shared ``--workload`` (+ ASE/gRASPA options) to *parser*."""
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOADS),
        default="thermo",
        help="Which tool to exercise (default: thermo = original MACE screen).",
    )
    parser.add_argument(
        "--calculator",
        default="mace_mp",
        help="ASE calculator for --workload ase (e.g. mace_mp, emt, tblite).",
    )
    parser.add_argument(
        "--driver",
        default="thermo",
        help="ASE driver for --workload ase (energy/opt/vib/thermo).",
    )
    parser.add_argument(
        "--adsorbate",
        default="H2O",
        help="gRASPA adsorbate for --workload graspa (only 'H2O' supported).",
    )
    parser.add_argument(
        "--graspa-cifs",
        nargs="+",
        default=None,
        help="CIF paths (remote-reachable) for --workload graspa.",
    )
    parser.add_argument(
        "--model-name",
        default="uma-s-1p1",
        help="FairChem/UMA model for --workload fairchem (e.g. uma-s-1p1, uma-m-1).",
    )
    parser.add_argument(
        "--pacmof2-cifs",
        nargs="+",
        default=None,
        help="CIF paths (remote-reachable) for --workload pacmof2.",
    )
    parser.add_argument(
        "--net-charge",
        type=float,
        default=0,
        help="Target net charge for --workload pacmof2 (default: 0).",
    )


def abort_if_graspa_unsupported(workload: str, backend_name: str) -> None:
    """Exit non-zero when gRASPA is requested on a non-HPC backend."""
    if workload == "graspa" and backend_name not in GRASPA_HPC_BACKENDS:
        print(
            "ERROR: gRASPA requires the HPC SYCL binary (see "
            "chemgraph.tools.graspa_core) and cannot run on the "
            f"{backend_name!r} backend. Use one of: "
            f"{sorted(GRASPA_HPC_BACKENDS)} on HPC."
        )
        sys.exit(2)


def resolve_items(workload: str, *, molecules: list[str], cifs: list[str] | None) -> list[str]:
    """Return the per-item list for *workload* (molecules or CIF paths).

    ``graspa`` and ``pacmof2`` are CIF-based; the caller passes the workload's
    ``--graspa-cifs`` / ``--pacmof2-cifs`` value as *cifs*.
    """
    if workload in _CIF_WORKLOADS:
        if not cifs:
            flag = "--pacmof2-cifs" if workload == "pacmof2" else "--graspa-cifs"
            print(f"ERROR: --workload {workload} requires {flag} <CIF> [<CIF> ...].")
            sys.exit(2)
        return list(cifs)
    return list(molecules)
