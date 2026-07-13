#!/usr/bin/env python
"""Direct ParslBackend demo running gRASPA (GCMC) over a directory of CIFs.

Must run inside a PBS interactive allocation on Polaris/Aurora/Crux::

    qsub -I -A <proj> -l select=1 -l walltime=01:00:00 -q debug \\
        -l filesystems=home:flare       # Aurora example

    module load frameworks              # or `module load conda` on Polaris
    source <venv>/bin/activate
    export COMPUTE_SYSTEM=aurora
    cd <repo>
    python scripts/demo/demo_graspa_parsl_direct.py \\
        --cif-dir /path/to/cifs --n-cycles 1000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_chemistry import (
    print_summary,
    submit_and_collect,
    write_csv,
)


def _abort(msg: str) -> None:
    print(f"[ABORT] {msg}")
    sys.exit(2)


def _collect_cifs(cif_dir: str) -> list[str]:
    root = Path(cif_dir).expanduser().resolve()
    if not root.is_dir():
        _abort(f"--cif-dir {root} is not a directory.")
    cifs = sorted(str(p) for p in root.glob("*.cif"))
    if not cifs:
        _abort(f"No .cif files found in {root}.")
    return cifs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cif-dir", required=True,
                        help="Directory containing .cif files (node-reachable / shared FS).")
    parser.add_argument("--system", default=os.environ.get("COMPUTE_SYSTEM"),
                        help="polaris | aurora | crux (default: $COMPUTE_SYSTEM)")
    parser.add_argument("--adsorbate", default="H2O",
                        help="Adsorbate (only 'H2O' supported today).")
    parser.add_argument("--temperature", type=float, default=298.15,
                        help="Temperature in K.")
    parser.add_argument("--pressure", type=float, default=101325.0,
                        help="Pressure in Pa.")
    parser.add_argument("--n-cycles", type=int, default=10000,
                        help="gRASPA GCMC cycles (both init + production).")
    parser.add_argument("--output-dir", default="demo_graspa_parsl_out")
    parser.add_argument("--run-dir", default=None,
                        help="Parsl run_dir (default: $PBS_O_WORKDIR/parsl_demo_runs).")
    parser.add_argument("--timeout", type=float, default=6000.0)
    args = parser.parse_args()

    pbs_nodefile = os.environ.get("PBS_NODEFILE")
    if not pbs_nodefile or not Path(pbs_nodefile).is_file():
        _abort("PBS_NODEFILE not set or missing. Run inside `qsub -I`.")
    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora", "crux"):
        _abort(f"Unsupported --system: {system!r}")

    cif_paths = _collect_cifs(args.cif_dir)

    run_dir = args.run_dir or os.environ.get("PBS_O_WORKDIR")
    if run_dir:
        run_dir = str(Path(run_dir) / "parsl_demo_runs")
    else:
        run_dir = str(Path.cwd() / "parsl_demo_runs")
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    print(f"system={system}  adsorbate={args.adsorbate}  T={args.temperature} K  "
          f"P={args.pressure} Pa  n_cycles={args.n_cycles}")
    print(f"cif_dir={args.cif_dir}  ({len(cif_paths)} CIFs)")
    print(f"run_dir={run_dir}")

    from chemgraph.execution.config import get_backend

    backend = get_backend(backend_name="parsl", system=system, run_dir=run_dir)
    try:
        results = submit_and_collect(
            backend,
            items=cif_paths,
            device="cpu",  # gRASPA-SYCL binary picks its own device
            output_dir=args.output_dir,
            inline=False,
            workload="graspa",
            adsorbate=args.adsorbate,
            temperature=args.temperature,
            pressure=args.pressure,
            n_cycles=args.n_cycles,
            timeout=args.timeout,
        )
    finally:
        backend.shutdown()

    csv_path = write_csv(results, Path(args.output_dir) / "demo_graspa_parsl.csv")
    print_summary(
        results,
        title=f"Parsl gRASPA GCMC screen (system={system}, adsorbate={args.adsorbate})",
        workload="graspa",
    )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
