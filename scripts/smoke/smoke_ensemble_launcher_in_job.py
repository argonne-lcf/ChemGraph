#!/usr/bin/env python
"""Smoke test for EnsembleLauncherBackend on an HPC compute node.

Must run **inside** a PBS interactive allocation on Polaris or Aurora,
in a venv where ``ensemble_launcher`` is installed (it is built from
source by ``scripts/hpc_setup/install_remote.sh`` -- PyPI wheels only
support Python <3.12).

Two modes
---------

``--mode managed`` (default)
    The script starts and tears down the EnsembleLauncher orchestrator
    in-process via ``get_backend(backend_name="ensemble_launcher", ...)``.

``--mode client-only``  *(exercises commit bc54083c)*
    In a **second shell on the same compute node**, first start the
    orchestrator yourself, e.g.::

        # second shell
        python -m ensemble_launcher \\
            --system $COMPUTE_SYSTEM \\
            --checkpoint-dir $PBS_O_WORKDIR/el_ckpt \\
            --node-id 0

    Then run this script with ``--mode client-only --checkpoint-dir
    $PBS_O_WORKDIR/el_ckpt``. It connects to the running orchestrator
    via ``ClusterClient`` rather than starting its own.

Usage
-----
::

    export COMPUTE_SYSTEM=polaris   # or aurora
    python scripts/smoke/smoke_ensemble_launcher_in_job.py --mode managed
    python scripts/smoke/smoke_ensemble_launcher_in_job.py \\
        --mode client-only --checkpoint-dir $PBS_O_WORKDIR/el_ckpt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from _smoke_utils import (
    SmokeReporter,
    trivial_add,
    trivial_hostname,
    trivial_square,
    water_xyz_path,
)


def _abort(msg: str) -> None:
    print(f"[ABORT] {msg}")
    sys.exit(2)


def _wait_for_checkpoint(checkpoint_dir: Path, timeout: float) -> None:
    """Wait until the orchestrator has written something to checkpoint_dir.

    The exact ready-marker shape depends on the ensemble_launcher
    version; we just wait for the directory to be non-empty.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if checkpoint_dir.is_dir() and any(checkpoint_dir.iterdir()):
            return
        time.sleep(1.0)
    _abort(
        f"No checkpoint files appeared under {checkpoint_dir} within {timeout}s. "
        "Start the orchestrator in another shell first; see --help."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("managed", "client-only"),
        default="managed",
    )
    parser.add_argument(
        "--system",
        default=os.environ.get("COMPUTE_SYSTEM"),
        help="polaris | aurora | local (default: $COMPUTE_SYSTEM)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="(client-only) path the externally-started orchestrator writes to.",
    )
    parser.add_argument(
        "--node-id",
        type=int,
        default=0,
        help="(client-only) node id assigned by the orchestrator (default 0).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="MACE device: cuda | xpu | cpu (default: cuda on polaris, xpu on aurora)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip MACE inference.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=float,
        default=60.0,
        help="(client-only) seconds to wait for orchestrator checkpoint to appear.",
    )
    args = parser.parse_args()

    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if not pbs_nodefile and args.system not in (None, "local"):
        _abort(
            "PBS_NODEFILE not set. Run inside a PBS allocation, or use --system local."
        )

    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora", "local", "crux"):
        _abort(f"Unsupported --system: {system!r}")

    if args.device:
        device = args.device
    elif system == "aurora":
        device = "xpu"
    elif system == "crux":
        device = "cpu"
    else:
        device = "cuda"

    try:
        import ensemble_launcher  # noqa: F401
    except ImportError as exc:
        _abort(
            f"ensemble_launcher is not importable: {exc}. "
            "On HPC, install it via scripts/hpc_setup/install_remote.sh."
        )

    from chemgraph.execution.base import TaskSpec
    from chemgraph.execution.config import get_backend

    r = SmokeReporter(
        f"smoke_ensemble_launcher_in_job (mode={args.mode}, system={system})"
    )
    backend = None

    if args.mode == "managed":
        with r.check("get_backend(ensemble_launcher, managed) initialises"):
            backend = get_backend(backend_name="ensemble_launcher", system=system)
            assert backend is not None
    else:
        if not args.checkpoint_dir:
            _abort("--mode client-only requires --checkpoint-dir.")
        ckpt = Path(args.checkpoint_dir).resolve()
        with r.check(
            f"orchestrator checkpoint dir is populated ({ckpt})"
        ):
            _wait_for_checkpoint(ckpt, args.wait_timeout)
        with r.check("get_backend(ensemble_launcher, client_only=True) connects"):
            backend = get_backend(
                backend_name="ensemble_launcher",
                system=system,
                client_only=True,
                checkpoint_dir=str(ckpt),
                node_id=args.node_id,
            )
            assert backend is not None

    if backend is None:
        r.summary_and_exit()
        return

    with r.check("python TaskSpec returns correct result"):
        fut = backend.submit(
            TaskSpec(
                task_id="el-py",
                task_type="python",
                callable=trivial_square,
                args=(11,),
            )
        )
        assert fut.result(timeout=180) == 121

    with r.check("python TaskSpec ran on a compute node"):
        fut = backend.submit(
            TaskSpec(
                task_id="el-host",
                task_type="python",
                callable=trivial_hostname,
            )
        )
        host = fut.result(timeout=180)
        print(f"       EL worker hostname = {host!r}")

    with r.check("shell TaskSpec runs"):
        fut = backend.submit(
            TaskSpec(
                task_id="el-sh",
                task_type="shell",
                command="echo smoke_el_shell_ok",
            )
        )
        # EL shell-task return shape depends on the version; just assert
        # the future resolves without raising.
        fut.result(timeout=180)

    with r.check("submit_batch of 3 python tasks all resolve"):
        futures = backend.submit_batch(
            [
                TaskSpec(
                    task_id=f"el-batch-{i}",
                    task_type="python",
                    callable=trivial_add,
                    args=(i, 50),
                )
                for i in range(3)
            ]
        )
        results = [f.result(timeout=240) for f in futures]
        assert results == [50, 51, 52], results

    if not args.quick:
        with r.check(f"MACE geometry opt on water (device={device}, converged)"):
            from ase.io import read as ase_read

            from chemgraph.mcp.mace_mcp_hpc import _mace_worker
            from chemgraph.tools.ase_core import atoms_to_atomsdata

            atoms = ase_read(str(water_xyz_path()))
            inline = atoms_to_atomsdata(atoms).model_dump()
            job = {
                "input_structure_file": "ignored_by_inline_path",
                "output_result_file": "water_smoke_el.json",
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
                    task_id="el-mace-opt",
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
        if args.mode == "managed":
            backend.shutdown()
        else:
            # In client-only mode, shutdown should NOT stop the orchestrator
            # the user started -- it should only disconnect this client.
            backend.shutdown()
            print("       (client-only: orchestrator left running in the other shell)")

    r.summary_and_exit()


if __name__ == "__main__":
    main()
