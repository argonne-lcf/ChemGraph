"""Subprocess entry point for one MACE calculation.

Invoked by :func:`chemgraph.mcp.mace_worker._mace_worker` as a fresh
Python interpreter to dodge a silent worker-hang seen when MACE runs
directly inside a Parsl ``process_worker_pool.py`` worker on Aurora.

CLI:
    python -m chemgraph.mcp._mace_subrunner <job.json> <result.json>

Reads *job.json*, runs the in-proc MACE worker, writes the result dict
to *result.json*. Exits non-zero on any uncaught exception.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: python -m chemgraph.mcp._mace_subrunner <job.json> <result.json>",
            file=sys.stderr,
        )
        return 2

    job_path, result_path = sys.argv[1], sys.argv[2]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)

    from chemgraph.mcp.mace_worker import _mace_worker_inproc

    result = _mace_worker_inproc(job)

    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, default=str)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
