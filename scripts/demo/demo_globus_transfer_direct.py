#!/usr/bin/env python
"""Direct Globus Transfer + Globus Compute demo.

Stages the 5 .xyz fixtures to a remote HPC collection via Globus
Transfer, then runs MACE ``driver="thermo"`` on each pre-staged file
through Globus Compute. Workers read the structures from the HPC
filesystem (remote-path mode), not embedded inline -- this exercises
``mace_mcp_hpc._mace_worker``'s ``remote_structure_file`` branch
(`mace_mcp_hpc.py:92-99`).

Prereq env vars::

    export GLOBUS_COMPUTE_ENDPOINT_ID=...
    export GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID=...        # laptop GCP collection
    export GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID=...   # HPC collection
    export GLOBUS_TRANSFER_DESTINATION_BASE_PATH=/eagle/projects/MyProj/staging

First run prompts for Globus OAuth; the token caches at
``~/.globus/chemgraph_transfer_tokens.json``.

Run::

    python scripts/demo/demo_globus_transfer_direct.py
    python scripts/demo/demo_globus_transfer_direct.py --device xpu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_chemistry import (
    MOLECULE_NAMES,
    _extract_properties,
    molecule_xyz_path,
    print_summary,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="demo_globus_transfer_out")
    parser.add_argument("--molecules", nargs="+", default=MOLECULE_NAMES)
    parser.add_argument("--device", default=os.environ.get("CG_DEMO_DEVICE", "cuda"))
    parser.add_argument(
        "--amqp-port",
        type=int,
        default=int(os.environ.get("CG_AMQP_PORT", "0")) or None,
    )
    parser.add_argument(
        "--transfer-timeout",
        type=float,
        default=6000.0,
        help="Seconds to wait for the Globus Transfer task (default 6000).",
    )
    parser.add_argument(
        "--compute-timeout",
        type=float,
        default=6000.0,
        help="Seconds to wait for each MACE thermo task (default 6000).",
    )
    args = parser.parse_args()

    required = (
        "GLOBUS_COMPUTE_ENDPOINT_ID",
        "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
    )
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}")
        sys.exit(2)

    from chemgraph.execution.base import TaskSpec
    from chemgraph.execution.config import get_backend, get_transfer_manager
    from chemgraph.mcp.mace_mcp_hpc import _mace_worker

    # ── 1. Stage all 5 .xyz files to the remote HPC collection ─────────
    print("\n[1/3] Submitting Globus Transfer for fixtures...")
    tm = get_transfer_manager()
    if tm is None:
        print("ERROR: get_transfer_manager() returned None.")
        sys.exit(2)

    local_paths = [str(molecule_xyz_path(n)) for n in args.molecules]
    transfer = tm.transfer_files(
        local_paths=local_paths,
        label=f"chemgraph-demo-{int(time.time())}",
    )
    print(f"      task_id   = {transfer.task_id}")
    print(f"      remote_dir = {transfer.remote_directory}")
    print(f"      waiting up to {args.transfer_timeout}s for SUCCEEDED...")
    status = tm.wait_for_transfer(
        transfer.task_id, timeout=args.transfer_timeout, poll_interval=5
    )
    if status.get("status") != "SUCCEEDED":
        print(f"ERROR: transfer did not succeed: {status}")
        sys.exit(1)
    print(
        f"      done: {status['files_transferred']}/{status['files']} files, "
        f"{status['bytes_transferred']} bytes"
    )

    # ── 2. Submit one MACE thermo task per pre-staged file ─────────────
    print(f"\n[2/3] Dispatching {len(args.molecules)} MACE thermo jobs via Globus Compute...")
    backend_kwargs = {}
    if args.amqp_port:
        backend_kwargs["amqp_port"] = args.amqp_port
    backend = get_backend(backend_name="globus_compute", **backend_kwargs)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    tasks = []
    for name in args.molecules:
        remote_xyz = f"{transfer.remote_directory}/{name}.xyz"
        job = {
            # input_structure_file is ignored when remote_structure_file is set
            # (mace_mcp_hpc._mace_worker:92-99 overrides it). Pass a sentinel.
            "input_structure_file": f"remote::{name}",
            "remote_structure_file": remote_xyz,
            "output_result_file": f"{name}_thermo.json",
            "driver": "thermo",
            "model": "medium-mpa-0",
            "device": args.device,
            "temperature": 298.15,
            "pressure": 101325.0,
            "fmax": 0.01,
            "steps": 200,
            "optimizer": "lbfgs",
        }
        jobs.append(job)
        tasks.append(
            TaskSpec(
                task_id=f"demo-tr-{name}",
                task_type="python",
                callable=_mace_worker,
                kwargs={"job": job},
            )
        )

    futures = backend.submit_batch(tasks)

    results = []
    try:
        for name, job, fut in zip(args.molecules, jobs, futures):
            print(f"      waiting on {name}...", flush=True)
            raw = fut.result(timeout=args.compute_timeout)
            if not isinstance(raw, dict) or raw.get("status") != "success":
                raise RuntimeError(f"{name}: backend returned {raw!r}")
            # Remote-path mode: full_output is NOT attached (only inline triggers
            # the JSON round-trip). Convergence + thermo cannot be read here
            # without staging the JSON back -- see the note in the summary table.
            results.append(_extract_properties(name, raw, job, inline=True))
    finally:
        backend.shutdown()

    # ── 3. Report ──────────────────────────────────────────────────────
    print(f"\n[3/3] Results (remote-path mode -- full JSON stays on the HPC):")
    print_summary(
        results,
        title=f"Globus Transfer + Compute thermo screen (device={args.device})",
    )
    csv_path = write_csv(results, output_dir / "demo_globus_transfer.csv")
    print(f"CSV (per-call status; thermo values blank in remote-path mode): {csv_path}")
    print(
        f"\nNote: workers wrote full JSON results under {transfer.remote_directory} "
        f"on the HPC. To pull them back, you can run another Globus Transfer "
        f"job in the reverse direction."
    )


if __name__ == "__main__":
    main()
