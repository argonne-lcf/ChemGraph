#!/usr/bin/env python
"""Direct GlobusComputeBackend demo: thermo screen on a remote HPC endpoint.

Submits 5 MACE ``driver="thermo"`` jobs to a Globus Compute endpoint
(Polaris/Aurora/etc.) and gathers results back to the laptop. The
structures are embedded inline (``inline=True``) so the workers don't
need to read anything from the laptop's filesystem.

Prereq env vars::

    export GLOBUS_COMPUTE_ENDPOINT_ID="<uuid>"      # required
    export COMPUTE_SYSTEM=polaris                    # optional, for logging
    # export CG_AMQP_PORT=443                       # if 5671 blocked (Aurora)

Run::

    python scripts/demo/demo_globus_compute_direct.py
    python scripts/demo/demo_globus_compute_direct.py --device xpu   # Aurora
    python scripts/demo/demo_globus_compute_direct.py --molecules water methane
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="demo_globus_compute_out")
    parser.add_argument("--molecules", nargs="+", default=MOLECULE_NAMES)
    parser.add_argument(
        "--device",
        default=os.environ.get("CG_DEMO_DEVICE", "cuda"),
        help="MACE/ASE device on the remote endpoint (default: cuda; use xpu on Aurora)",
    )
    parser.add_argument(
        "--amqp-port",
        type=int,
        default=int(os.environ.get("CG_AMQP_PORT", "0")) or None,
        help="Override AMQP port (set to 443 if 5671 is blocked, e.g. Aurora)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=6000.0,
        help="Per-task timeout in seconds (default 6000)",
    )
    add_workload_args(parser)
    args = parser.parse_args()

    if not os.environ.get("GLOBUS_COMPUTE_ENDPOINT_ID"):
        print("ERROR: export GLOBUS_COMPUTE_ENDPOINT_ID=<uuid> first.")
        sys.exit(2)

    abort_if_graspa_unsupported(args.workload, "globus_compute")
    items = resolve_items(args.workload, molecules=args.molecules, cifs=args.graspa_cifs)
    # MACE/ASE embed structures inline (no shared FS on the endpoint); gRASPA
    # reads pre-staged CIFs from the remote filesystem, so it is not inline.
    inline = args.workload != "graspa"

    from chemgraph.execution.config import get_backend

    backend_kwargs: dict = {}
    if args.amqp_port:
        backend_kwargs["amqp_port"] = args.amqp_port

    backend = get_backend(backend_name="globus_compute", **backend_kwargs)
    try:
        results = submit_and_collect(
            backend,
            items=items,
            device=args.device,
            output_dir=args.output_dir,
            inline=inline,
            workload=args.workload,
            calculator=args.calculator,
            driver=args.driver,
            adsorbate=args.adsorbate,
            timeout=args.timeout,
        )
    finally:
        backend.shutdown()

    csv_path = write_csv(results, Path(args.output_dir) / "demo_globus_compute.csv")
    print_summary(
        results,
        title=(
            f"Globus Compute {args.workload} screen "
            f"(system={os.environ.get('COMPUTE_SYSTEM', '?')}, device={args.device})"
        ),
        workload=args.workload,
    )
    print(f"CSV written to: {csv_path}")


if __name__ == "__main__":
    main()
