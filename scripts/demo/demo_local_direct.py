#!/usr/bin/env python
"""Direct LocalBackend demo: thermochemistry screen of 5 small molecules.

Runs entirely on the laptop, no LLM, no HPC. Submits 5 MACE
``driver="thermo"`` jobs to a ``LocalBackend`` ProcessPoolExecutor,
gathers the results, prints a property table, and writes a CSV.

Run::

    python scripts/demo/demo_local_direct.py
    python scripts/demo/demo_local_direct.py --output-dir /tmp/cg_demo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make _demo_chemistry importable when run from any cwd.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="demo_local_out",
        help="Where per-molecule JSON + CSV land (default: ./demo_local_out)",
    )
    parser.add_argument(
        "--molecules",
        nargs="+",
        default=MOLECULE_NAMES,
        help=f"Subset to run (default: {MOLECULE_NAMES})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="MACE/ASE device (default: cpu; local Mac/CPU)",
    )
    add_workload_args(parser)
    args = parser.parse_args()

    abort_if_graspa_unsupported(args.workload, "local")
    items = resolve_items(args.workload, molecules=args.molecules, cifs=args.graspa_cifs)

    from chemgraph.execution.config import get_backend

    backend = get_backend(backend_name="local", system="local")
    try:
        results = submit_and_collect(
            backend,
            items=items,
            device=args.device,
            output_dir=args.output_dir,
            inline=False,
            workload=args.workload,
            calculator=args.calculator,
            driver=args.driver,
            adsorbate=args.adsorbate,
            timeout=1200,
        )
    finally:
        backend.shutdown()

    csv_path = write_csv(results, Path(args.output_dir) / "demo_local.csv")
    print_summary(
        results,
        title=f"Local backend {args.workload} screen ({args.device})",
        workload=args.workload,
    )
    print(f"CSV written to: {csv_path}")
    print(f"Per-item output written under: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
