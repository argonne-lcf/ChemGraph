#!/usr/bin/env python
"""Smoke test for the LocalBackend.

Drives the production execution layer end-to-end on the laptop with no
HPC and no credentials. Exits 0 on success, nonzero on any failure.

Checks
------
1. ``get_backend(backend_name="local")`` initialises cleanly.
2. Python TaskSpec round-trip (callable returns correct result).
3. Shell TaskSpec round-trip (exit code 0).
4. ``submit_batch`` of three tasks all resolve.
5. ``JobTracker`` register_batch / get_status / get_results round-trip.
6. MACE worker path: build a job dict for ``water.xyz`` and submit it to
   the local backend exactly as ``mace_mcp_hpc._mace_transport_hook`` would.

Run::

    python scripts/smoke/smoke_local.py
    python scripts/smoke/smoke_local.py --quick   # skip the MACE check
"""

from __future__ import annotations

import argparse

from _smoke_utils import (
    SmokeReporter,
    trivial_add,
    trivial_square,
    water_xyz_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip MACE inference (saves ~30s on first run downloading the model).",
    )
    args = parser.parse_args()

    from chemgraph.execution.base import TaskSpec
    from chemgraph.execution.config import get_backend

    r = SmokeReporter("smoke_local")
    backend = None

    with r.check("get_backend(local) initialises"):
        backend = get_backend(backend_name="local", system="local")
        assert backend is not None, "backend is None"

    if backend is None:
        r.summary_and_exit()
        return

    with r.check("python TaskSpec returns correct result"):
        fut = backend.submit(
            TaskSpec(
                task_id="py-1",
                task_type="python",
                callable=trivial_square,
                args=(7,),
            )
        )
        result = fut.result(timeout=30)
        assert result == 49, f"expected 49, got {result!r}"

    with r.check("shell TaskSpec exits 0"):
        fut = backend.submit(
            TaskSpec(
                task_id="sh-1",
                task_type="shell",
                command="echo smoke_local_shell_ok",
            )
        )
        rc = fut.result(timeout=30)
        assert rc == 0, f"expected exit 0, got {rc!r}"

    with r.check("submit_batch of 3 python tasks resolve"):
        futures = backend.submit_batch(
            [
                TaskSpec(
                    task_id=f"batch-{i}",
                    task_type="python",
                    callable=trivial_add,
                    args=(i, i + 1),
                )
                for i in range(3)
            ]
        )
        results = [f.result(timeout=30) for f in futures]
        assert results == [1, 3, 5], f"expected [1,3,5], got {results}"

    with r.check("JobTracker register_batch / get_results round-trip"):
        from chemgraph.execution.job_tracker import JobTracker

        tracker = JobTracker()
        fut = backend.submit(
            TaskSpec(
                task_id="tracked-1",
                task_type="python",
                callable=trivial_square,
                args=(6,),
            )
        )
        batch_id = tracker.register_batch(
            tool_name="smoke_local",
            pending_tasks=[({"task_id": "tracked-1"}, fut)],
        )
        # Block on the future then ask the tracker for results.
        fut.result(timeout=30)
        out = tracker.get_results(batch_id)
        assert out["status"] == "completed", f"status={out.get('status')}"
        assert out["results"][0]["result"] == 36, out["results"]

    if not args.quick:
        with r.check("MACE geometry opt: water.xyz on local backend (converged)"):
            import json

            from chemgraph.mcp.mace_mcp_hpc import _mace_worker

            xyz = water_xyz_path()
            assert xyz.is_file(), f"fixture missing: {xyz}"
            out_json = xyz.parent / "water_smoke_output.json"
            job = {
                "input_structure_file": str(xyz),
                "output_result_file": str(out_json),
                "driver": "opt",
                "model": "medium-mpa-0",
                "device": "cpu",
                "temperature": 298.15,
                "pressure": 101325.0,
                "fmax": 0.01,
                "steps": 100,
                "optimizer": "lbfgs",
            }
            # Submit through the backend (not in-process) to prove the
            # submission pipeline serializes the worker callable and the
            # arg dict correctly.
            fut = backend.submit(
                TaskSpec(
                    task_id="mace-water-opt",
                    task_type="python",
                    callable=_mace_worker,
                    kwargs={"job": job},
                )
            )
            # First MACE run downloads the model; allow generous timeout.
            mace_out = fut.result(timeout=600)
            assert isinstance(mace_out, dict), f"non-dict result: {type(mace_out)}"
            assert mace_out.get("status") == "success", f"opt failed: {mace_out}"
            energy = next(
                (mace_out[k] for k in ("single_point_energy", "energy", "final_energy") if k in mace_out),
                None,
            )
            assert energy is not None, f"no energy in result keys={list(mace_out)}"
            assert energy < 0, f"water energy should be negative, got {energy}"

            assert out_json.is_file(), f"opt output JSON not written: {out_json}"
            with open(out_json) as fh:
                full = json.load(fh)
            assert full.get("converged") is True, f"opt did not converge: {full.get('converged')!r}"
            assert full.get("success") is True, f"opt success=False: {full}"
            print(
                f"       water opt energy = {energy:.6f} eV  "
                f"(converged={full.get('converged')}, wall={full.get('wall_time')}s)"
            )

    with r.check("backend.shutdown() is clean"):
        backend.shutdown()

    r.summary_and_exit()


if __name__ == "__main__":
    main()
