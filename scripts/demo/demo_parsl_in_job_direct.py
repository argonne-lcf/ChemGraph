#!/usr/bin/env python
"""Direct ParslBackend demo on an HPC compute node: 5-molecule thermo screen.

Must run inside a PBS interactive allocation on Polaris or Aurora::

    # Polaris
    qsub -I -A <proj> -l select=1 -l walltime=01:00:00 -q debug -l filesystems=home:eagle
    # Aurora
    qsub -I -A <proj> -l select=1,walltime=01:00:00 -q debug -l filesystems=home:flare

Inside the allocation::

    module load conda                   # or `module load frameworks` on Aurora
    source <venv>/bin/activate
    export COMPUTE_SYSTEM=polaris        # or aurora
    cd <repo>
    python scripts/demo/demo_parsl_in_job_direct.py
    python scripts/demo/demo_parsl_in_job_direct.py --device xpu   # Aurora
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_chemistry import (
    MOLECULE_NAMES,
    print_summary,
    submit_and_collect,
    write_csv,
)


def _abort(msg: str) -> None:
    print(f"[ABORT] {msg}")
    sys.exit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        default=os.environ.get("COMPUTE_SYSTEM"),
        help="polaris | aurora (default: $COMPUTE_SYSTEM)",
    )
    parser.add_argument("--device", default=None, help="cuda (Polaris) | xpu (Aurora)")
    parser.add_argument("--output-dir", default="demo_parsl_out")
    parser.add_argument("--molecules", nargs="+", default=MOLECULE_NAMES)
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Parsl run_dir (default: $PBS_O_WORKDIR/parsl_demo_runs or ./parsl_demo_runs).",
    )
    parser.add_argument("--timeout", type=float, default=6000.0)
    args = parser.parse_args()

    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if not pbs_nodefile or not Path(pbs_nodefile).is_file():
        _abort("PBS_NODEFILE not set or missing. Run inside `qsub -I`.")
    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora"):
        _abort(f"Unsupported --system: {system!r}")
    device = args.device or ("xpu" if system == "aurora" else "cuda")

    run_dir = args.run_dir or os.environ.get("PBS_O_WORKDIR")
    if run_dir:
        run_dir = str(Path(run_dir) / "parsl_demo_runs")
    else:
        run_dir = str(Path.cwd() / "parsl_demo_runs")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    print(f"system={system}  device={device}  run_dir={run_dir}")

    from chemgraph.execution.config import get_backend

    backend = get_backend(backend_name="parsl", system=system, run_dir=run_dir)
    try:
        results = submit_and_collect(
            backend,
            molecule_names=args.molecules,
            device=device,
            output_dir=args.output_dir,
            inline=False,
            timeout=args.timeout,
        )
    finally:
        backend.shutdown()

    csv_path = write_csv(results, Path(args.output_dir) / "demo_parsl.csv")
    print_summary(results, title=f"Parsl thermo screen (system={system}, device={device})")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
