#!/usr/bin/env python
"""Smoke test for GlobusTransferManager (+ optional MCP integration).

Exercises the production transfer layer from the laptop. Exits 0 on
success, nonzero on any failure.

Prereqs (env vars)
------------------
- GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID       -- local Globus collection UUID
- GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID  -- HPC collection UUID
- GLOBUS_TRANSFER_DESTINATION_BASE_PATH    -- e.g. /eagle/projects/MyProj/staging
- (for --with-mcp): GLOBUS_COMPUTE_ENDPOINT_ID and HPC venv with MACE

First run triggers a Globus OAuth flow. Token caches at
~/.globus/chemgraph_transfer_tokens.json.

Run::

    python scripts/smoke/smoke_globus_transfer.py
    python scripts/smoke/smoke_globus_transfer.py --keep-remote   # don't delete after
    python scripts/smoke/smoke_globus_transfer.py --with-mcp      # also exercise MCP ensemble in remote mode
"""

from __future__ import annotations

import argparse
import os
import time

from _smoke_utils import SmokeReporter, require_env, water_xyz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-mcp",
        action="store_true",
        help="Also exercise mace_mcp_hpc.run_mace_ensemble(remote_structure_directory=...).",
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="Don't attempt to delete the staged remote directory at the end.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=6000.0,
        help="Per-transfer timeout in seconds (default 6000).",
    )
    args = parser.parse_args()

    require_env(
        "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
    )
    if args.with_mcp:
        require_env("GLOBUS_COMPUTE_ENDPOINT_ID")

    from chemgraph.execution.config import get_transfer_manager

    r = SmokeReporter("smoke_globus_transfer")
    mgr = None
    transfer_result = None

    with r.check("get_transfer_manager() returns a configured manager"):
        mgr = get_transfer_manager()
        assert mgr is not None, (
            "get_transfer_manager returned None -- check env vars are exported."
        )

    if mgr is None:
        r.summary_and_exit()
        return

    with r.check("transfer_files(water.xyz) submits a Globus Transfer task"):
        xyz = water_xyz_path()
        assert xyz.is_file(), f"fixture missing: {xyz}"
        transfer_result = mgr.transfer_files(
            local_paths=[str(xyz)],
            label=f"chemgraph-smoke-{int(time.time())}",
        )
        assert transfer_result.task_id, "no task_id returned"
        print(f"       task_id = {transfer_result.task_id}")
        print(f"       remote_dir = {transfer_result.remote_directory}")

    with r.check(f"wait_for_transfer(timeout={args.timeout}s) reaches SUCCEEDED"):
        assert transfer_result is not None
        status = mgr.wait_for_transfer(
            transfer_result.task_id,
            timeout=args.timeout,
            poll_interval=5,
        )
        assert status.get("status") == "SUCCEEDED", f"final status: {status}"
        assert status.get("files_transferred", 0) >= 1, status
        print(
            f"       transferred {status['files_transferred']}/{status['files']} files, "
            f"{status['bytes_transferred']} bytes"
        )

    with r.check("check_transfer_status() returns SUCCEEDED for completed task"):
        assert transfer_result is not None
        status = mgr.check_transfer_status(transfer_result.task_id)
        assert status["status"] == "SUCCEEDED", status

    with r.check("list_remote_directory() finds the staged file"):
        assert transfer_result is not None
        entries = mgr.list_remote_directory(transfer_result.remote_directory)
        names = {e["name"] for e in entries}
        assert "water.xyz" in names, f"water.xyz not in {names!r}"
        size = next((e["size"] for e in entries if e["name"] == "water.xyz"), 0)
        print(f"       remote water.xyz size = {size} bytes")

    if args.with_mcp:
        with r.check("MCP run_mace_ensemble(remote_structure_directory=...) succeeds"):
            # Drive the MCP server's tool function directly (in-process) --
            # the heavy work is dispatched to Globus Compute by the
            # backend that mcp.init_backend() configured.
            from chemgraph.mcp.mace_mcp_hpc import (
                _expand_mace_ensemble,
                _mace_worker,
                mcp,
            )
            from chemgraph.execution.base import TaskSpec
            from chemgraph.schemas.mace_parsl_schema import (
                mace_input_schema_ensemble,
            )

            # Init the MCP server's backend (reads CHEMGRAPH_EXECUTION_BACKEND
            # / GLOBUS_COMPUTE_ENDPOINT_ID exactly like the prod server does).
            os.environ.setdefault("CHEMGRAPH_EXECUTION_BACKEND", "globus_compute")
            mcp.init_backend()
            try:
                params = mace_input_schema_ensemble(
                    remote_structure_directory=transfer_result.remote_directory,
                    output_result_file="water_smoke_tr.json",
                    driver="opt",
                    model="medium-mpa-0",
                    device=os.environ.get("CG_SMOKE_DEVICE", "cuda"),
                )
                jobs = _expand_mace_ensemble(params)
                assert jobs, "no jobs expanded from remote dir"
                assert all("remote_structure_file" in j for j in jobs), jobs[0]
                # Submit each job through the same backend the MCP server uses.
                futures = [
                    mcp._backend.submit(
                        TaskSpec(
                            task_id=f"tr-mace-opt-{i}",
                            task_type="python",
                            callable=_mace_worker,
                            kwargs={"job": j},
                        )
                    )
                    for i, j in enumerate(jobs)
                ]
                results = [f.result(timeout=6000) for f in futures]
                assert all(isinstance(r, dict) for r in results), results
                assert all(r.get("status") == "success" for r in results), [
                    r.get("status") for r in results
                ]
                energies = [
                    next(
                        (r[k] for k in ("single_point_energy", "energy", "final_energy") if k in r),
                        None,
                    )
                    for r in results
                ]
                assert all(e is not None and e < 0 for e in energies), results
                # Remote-path mode does NOT attach full_output (only the
                # inline-structure path does -- see mace_mcp_hpc._mace_worker
                # lines 127-131). Convergence can be verified after the fact
                # by reading the per-structure JSON on the remote filesystem
                # (e.g. via Globus Transfer back to the laptop) -- out of
                # scope for this smoke test.
                print(f"       remote MACE opt energies (eV): {energies}")
            finally:
                mcp.shutdown_backend()

    if not args.keep_remote and transfer_result is not None:
        print(
            f"\nNOTE: staged directory left in place at {transfer_result.remote_directory}\n"
            "      (the manager does not implement remote deletion). "
            "Clean it up manually if needed."
        )

    r.summary_and_exit()


if __name__ == "__main__":
    main()
