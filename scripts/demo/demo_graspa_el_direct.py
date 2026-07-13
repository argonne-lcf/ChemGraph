#!/usr/bin/env python
"""Direct EnsembleLauncherBackend demo running gRASPA (GCMC) over a CIF dir.

Must run inside ``qsub -I`` on Polaris/Aurora/Crux in a venv where
``ensemble_launcher`` is installed (built from source by
``scripts/hpc_setup/install_remote.sh``)::

    export COMPUTE_SYSTEM=aurora
    python scripts/demo/demo_graspa_el_direct.py \\
        --cif-dir /path/to/cifs --n-cycles 1000 --ppn 16
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
                        help="gRASPA GCMC cycles.")
    parser.add_argument("--output-dir", default="demo_graspa_el_out")
    parser.add_argument("--ppn", type=int, default=16,
                        help="Processes (cores) per node for each task.")
    parser.add_argument("--timeout", type=float, default=6000.0)
    args = parser.parse_args()

    if not os.environ.get("PBS_NODEFILE"):
        _abort("PBS_NODEFILE not set. Run inside `qsub -I`.")
    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora", "crux"):
        _abort(f"Unsupported --system: {system!r}")

    try:
        import ensemble_launcher  # noqa: F401
    except ImportError as exc:
        _abort(
            f"ensemble_launcher import failed: {exc}. "
            "Install via scripts/hpc_setup/install_remote.sh on HPC."
        )

    cif_paths = _collect_cifs(args.cif_dir)

    print(f"system={system}  adsorbate={args.adsorbate}  T={args.temperature} K  "
          f"P={args.pressure} Pa  n_cycles={args.n_cycles}  ppn={args.ppn}  mode=managed")
    print(f"cif_dir={args.cif_dir}  ({len(cif_paths)} CIFs)")

    from chemgraph.execution.config import get_backend

    backend = get_backend(backend_name="ensemble_launcher", system=system)
    try:
        results = submit_and_collect(
            backend,
            items=cif_paths,
            device="cpu",
            output_dir=args.output_dir,
            inline=False,
            workload="graspa",
            adsorbate=args.adsorbate,
            temperature=args.temperature,
            pressure=args.pressure,
            n_cycles=args.n_cycles,
            timeout=args.timeout,
            ppn=args.ppn,
        )
    finally:
        backend.shutdown()

    csv_path = write_csv(results, Path(args.output_dir) / "demo_graspa_el.csv")
    print_summary(
        results,
        title=f"EnsembleLauncher gRASPA GCMC screen (system={system}, adsorbate={args.adsorbate})",
        workload="graspa",
    )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
