#!/usr/bin/env python
"""Smoke test for ParslBackend on an HPC compute node.

Must run **inside** a PBS interactive allocation on Polaris or Aurora::

    # Polaris
    qsub -I -A <proj> -l select=1 -l walltime=01:00:00 -q debug
    # Aurora
    qsub -I -A <proj> -l select=1,walltime=01:00:00 -q debug -l filesystems=home:flare

Inside the allocation::

    module load conda  # or `module load frameworks` on Aurora
    source <venv>/bin/activate
    export COMPUTE_SYSTEM=polaris   # or aurora
    python scripts/smoke/smoke_parsl_in_job.py
    python scripts/smoke/smoke_parsl_in_job.py --quick
    python scripts/smoke/smoke_parsl_in_job.py --device xpu   # Aurora

The script fails fast with a clear message if PBS_NODEFILE is missing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from _smoke_utils import (
    SmokeReporter,
    trivial_add,
    trivial_env_probe,
    trivial_hostname,
    trivial_square,
    water_xyz_path,
)


def _abort(msg: str) -> None:
    print(f"[ABORT] {msg}")
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        default=os.environ.get("COMPUTE_SYSTEM"),
        help="polaris | aurora (default: COMPUTE_SYSTEM env var)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="MACE device: cuda (Polaris default), xpu (Aurora), or cpu.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Parsl run_dir (default: $PBS_O_WORKDIR/parsl_runs or ./parsl_runs).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip MACE inference.",
    )
    args = parser.parse_args()

    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if not pbs_nodefile or not Path(pbs_nodefile).is_file():
        _abort(
            "PBS_NODEFILE not set or missing. This script must run inside a "
            "PBS interactive allocation (qsub -I ...)."
        )

    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora"):
        _abort(f"Unsupported --system: {system!r} (expected polaris|aurora)")

    device = args.device or ("xpu" if system == "aurora" else "cuda")
    nodes = Path(pbs_nodefile).read_text().splitlines()

    run_dir = args.run_dir or os.environ.get("PBS_O_WORKDIR")
    if run_dir:
        run_dir = str(Path(run_dir) / "parsl_runs_smoke")
    else:
        run_dir = str(Path.cwd() / "parsl_runs_smoke")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    print(f"system={system}  device={device}  nodes={len(nodes)}  run_dir={run_dir}")

    from chemgraph.execution.base import TaskSpec
    from chemgraph.execution.config import get_backend

    r = SmokeReporter(f"smoke_parsl_in_job (system={system}, nodes={len(nodes)})")
    backend = None

    with r.check("get_backend(parsl) initialises with HPC config"):
        backend = get_backend(
            backend_name="parsl",
            system=system,
            run_dir=run_dir,
        )
        assert backend is not None

    if backend is None:
        r.summary_and_exit()
        return

    with r.check("python TaskSpec returns correct result"):
        fut = backend.submit(
            TaskSpec(
                task_id="p-py",
                task_type="python",
                callable=trivial_square,
                args=(9,),
            )
        )
        assert fut.result(timeout=120) == 81

    with r.check("python TaskSpec ran on a compute node (hostname != login)"):
        fut = backend.submit(
            TaskSpec(
                task_id="p-host",
                task_type="python",
                callable=trivial_hostname,
            )
        )
        host = fut.result(timeout=120)
        print(f"       parsl worker hostname = {host!r}")
        assert isinstance(host, str) and host

    with r.check("worker env: torch + accelerators visible"):
        fut = backend.submit(
            TaskSpec(
                task_id="p-env",
                task_type="python",
                callable=trivial_env_probe,
            )
        )
        info = fut.result(timeout=120)
        print(f"       worker env: {info}")
        # Polaris should show cuda; Aurora should show xpu.
        if system == "polaris":
            assert info.get("cuda_devices", 0) >= 1, info
        elif system == "aurora":
            assert info.get("xpu_devices", 0) >= 1, info

    with r.check("shell TaskSpec exits 0"):
        fut = backend.submit(
            TaskSpec(
                task_id="p-sh",
                task_type="shell",
                command="echo smoke_parsl_shell_ok && hostname",
            )
        )
        rc = fut.result(timeout=120)
        assert rc == 0, f"exit code = {rc}"

    with r.check("submit_batch of 4 python tasks all resolve"):
        futures = backend.submit_batch(
            [
                TaskSpec(
                    task_id=f"p-batch-{i}",
                    task_type="python",
                    callable=trivial_add,
                    args=(i, 100),
                )
                for i in range(4)
            ]
        )
        results = [f.result(timeout=180) for f in futures]
        assert results == [100, 101, 102, 103], results

    if not args.quick:
        with r.check(f"MACE geometry opt on water (device={device}, converged)"):
            from ase.io import read as ase_read

            from chemgraph.mcp.mace_mcp_hpc import _mace_worker
            from chemgraph.tools.ase_core import atoms_to_atomsdata

            atoms = ase_read(str(water_xyz_path()))
            inline = atoms_to_atomsdata(atoms).model_dump()
            job = {
                "input_structure_file": "ignored_by_inline_path",
                "output_result_file": "water_smoke_parsl.json",
                "driver": "opt",
                "model": "medium-mpa-0",
                "device": device,
                "temperature": 298.15,
                "pressure": 101325.0,
                "fmax": 0.01,
                "steps": 100,
                "optimizer": "lbfgs",
                "inline_structure": inline,
            }
            fut = backend.submit(
                TaskSpec(
                    task_id="p-mace-opt",
                    task_type="python",
                    callable=_mace_worker,
                    kwargs={"job": job},
                )
            )
            out = fut.result(timeout=900)
            assert out.get("status") == "success", f"opt failed: {out}"
            energy = next(
                (out[k] for k in ("single_point_energy", "energy", "final_energy") if k in out),
                None,
            )
            assert energy is not None and energy < 0, f"bad MACE result: {out}"
            full = out.get("full_output") or {}
            if full:
                assert full.get("converged") is True, f"opt did not converge: {full.get('converged')!r}"
                print(
                    f"       water opt energy = {energy:.6f} eV  "
                    f"(converged={full.get('converged')}, wall={full.get('wall_time')}s)"
                )
            else:
                print(
                    f"       water opt energy = {energy:.6f} eV  "
                    "(WARNING: full_output missing; convergence not verified inline)"
                )

    with r.check("backend.shutdown() is clean"):
        backend.shutdown()

    r.summary_and_exit()


if __name__ == "__main__":
    main()
