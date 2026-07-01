#!/usr/bin/env python
"""Direct EnsembleLauncherBackend demo on an HPC compute node.

5-molecule thermo screen via the EnsembleLauncher orchestrator,
managed mode (the backend starts and tears down the orchestrator
itself). Must run inside ``qsub -I`` on Polaris or Aurora, in a venv
where ``ensemble_launcher`` is installed (built from source by
``scripts/hpc_setup/install_remote.sh``).

Run::

    export COMPUTE_SYSTEM=polaris
    python scripts/demo/demo_ensemble_launcher_in_job_direct.py
    python scripts/demo/demo_ensemble_launcher_in_job_direct.py --device xpu
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_chemistry import (
    MOLECULE_NAMES,
    abort_if_graspa_unsupported,
    add_workload_args,
    print_summary,
    resolve_items,
    submit_and_collect,
    write_csv,
)


def _abort(msg: str) -> None:
    print(f"[ABORT] {msg}")
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", default=os.environ.get("COMPUTE_SYSTEM"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="demo_el_out")
    parser.add_argument("--molecules", nargs="+", default=MOLECULE_NAMES)
    parser.add_argument("--ppn", type=int, default=16,
                        help="Processes (cores) per node for each task")
    parser.add_argument("--timeout", type=float, default=6000.0)
    add_workload_args(parser)
    args = parser.parse_args()

    abort_if_graspa_unsupported(args.workload, "ensemble_launcher")

    if not os.environ.get("PBS_NODEFILE"):
        _abort("PBS_NODEFILE not set. Run inside `qsub -I`.")
    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora", "crux"):
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
            f"ensemble_launcher import failed: {exc}. "
            "Install via scripts/hpc_setup/install_remote.sh on HPC."
        )

    print(f"system={system}  device={device}  ppn={args.ppn}  mode=managed")

    from chemgraph.execution.config import get_backend

    items = resolve_items(args.workload, molecules=args.molecules, cifs=args.graspa_cifs)

    backend = get_backend(backend_name="ensemble_launcher", system=system)
    try:
        results = submit_and_collect(
            backend,
            items=items,
            device=device,
            output_dir=args.output_dir,
            inline=False,
            workload=args.workload,
            calculator=args.calculator,
            driver=args.driver,
            adsorbate=args.adsorbate,
            timeout=args.timeout,
            ppn=args.ppn,
        )
    finally:
        backend.shutdown()

    csv_path = write_csv(results, Path(args.output_dir) / "demo_el.csv")
    print_summary(
        results,
        title=f"EnsembleLauncher {args.workload} screen (system={system}, device={device})",
        workload=args.workload,
    )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
