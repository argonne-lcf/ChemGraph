#!/usr/bin/env python
"""Smoke test for the GlobusComputeBackend.

Drives the production execution layer against a live Globus Compute
endpoint. Exits 0 on success, nonzero on any failure.

Prereqs (env vars)
------------------
- GLOBUS_COMPUTE_ENDPOINT_ID  -- UUID printed by ``globus-compute-endpoint start``.
- (optional) COMPUTE_SYSTEM   -- "polaris" or "aurora" (used for logging only).

Run::

    export GLOBUS_COMPUTE_ENDPOINT_ID="<uuid>"
    python scripts/smoke/smoke_globus_compute.py
    python scripts/smoke/smoke_globus_compute.py --quick     # skip MACE
    python scripts/smoke/smoke_globus_compute.py --amqp 443  # firewalled networks
"""

from __future__ import annotations

import argparse
import os

from _smoke_utils import (
    SmokeReporter,
    require_env,
    trivial_add,
    trivial_env_probe,
    trivial_hostname,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip MACE inference (Globus model download on remote endpoint is slow on first run).",
    )
    parser.add_argument(
        "--amqp",
        type=int,
        default=None,
        help="AMQP port override. Set to 443 when outbound 5671 is blocked (Aurora).",
    )
    args = parser.parse_args()

    require_env("GLOBUS_COMPUTE_ENDPOINT_ID")

    from chemgraph.execution.base import TaskSpec
    from chemgraph.execution.config import get_backend

    backend_kwargs: dict = {}
    if args.amqp is not None:
        backend_kwargs["amqp_port"] = args.amqp

    r = SmokeReporter(
        f"smoke_globus_compute (system={os.environ.get('COMPUTE_SYSTEM', '?')}, "
        f"endpoint={os.environ['GLOBUS_COMPUTE_ENDPOINT_ID'][:8]}...)"
    )
    backend = None
    local_hostname = trivial_hostname()

    with r.check("get_backend(globus_compute) initialises"):
        backend = get_backend(backend_name="globus_compute", **backend_kwargs)
        assert backend is not None

    if backend is None:
        r.summary_and_exit()
        return

    with r.check("check_endpoint_status() reports online"):
        status = backend.check_endpoint_status()
        # The SDK returns either a dict like {"status": "online"} or a
        # string; both shapes count as healthy if "online" appears in the
        # repr. "error" status means we cannot reach the endpoint.
        s = status.get("status")
        assert s != "error", f"endpoint unreachable: {status}"
        s_repr = str(s).lower()
        assert "online" in s_repr or "ok" in s_repr or "running" in s_repr, (
            f"endpoint not online: {status}"
        )
        print(f"       endpoint status: {status}")

    with r.check("python TaskSpec (trivial_add) round-trips through Globus"):
        fut = backend.submit(
            TaskSpec(
                task_id="gc-add",
                task_type="python",
                callable=trivial_add,
                args=(40, 2),
            )
        )
        result = fut.result(timeout=300)
        assert result == 42, f"expected 42, got {result!r}"

    with r.check("python TaskSpec ran on the HPC node (hostname differs from laptop)"):
        fut = backend.submit(
            TaskSpec(
                task_id="gc-host",
                task_type="python",
                callable=trivial_hostname,
            )
        )
        remote_host = fut.result(timeout=300)
        assert isinstance(remote_host, str) and remote_host, "empty hostname"
        assert remote_host != local_hostname, (
            f"task ran on the laptop ({remote_host}), not the endpoint!"
        )
        print(f"       local={local_hostname!r}  remote={remote_host!r}")

    with r.check("env probe: torch + accelerators visible on worker"):
        fut = backend.submit(
            TaskSpec(
                task_id="gc-env",
                task_type="python",
                callable=trivial_env_probe,
            )
        )
        info = fut.result(timeout=300)
        assert isinstance(info, dict)
        print(f"       worker env: {info}")

    with r.check("shell TaskSpec returns SDK ShellResult"):
        fut = backend.submit(
            TaskSpec(
                task_id="gc-sh",
                task_type="shell",
                command="echo smoke_globus_compute_shell_ok && hostname",
            )
        )
        sh = fut.result(timeout=300)
        # ShellFunction returns a ShellResult object with .stdout
        stdout = getattr(sh, "stdout", str(sh))
        assert "smoke_globus_compute_shell_ok" in stdout, f"unexpected stdout: {stdout!r}"
        print(f"       remote shell stdout (truncated): {stdout[:120]!r}")

    with r.check("submit_batch of 3 python tasks all resolve"):
        futures = backend.submit_batch(
            [
                TaskSpec(
                    task_id=f"gc-batch-{i}",
                    task_type="python",
                    callable=trivial_add,
                    args=(i, 10),
                )
                for i in range(3)
            ]
        )
        results = [f.result(timeout=300) for f in futures]
        assert results == [10, 11, 12], f"expected [10,11,12], got {results}"

    if not args.quick:
        with r.check("MACE geometry opt on water runs on Globus Compute (converged)"):
            from chemgraph.mcp.mace_mcp_hpc import _mace_worker

            # Worker pulls the structure from its own filesystem.  Since
            # the laptop's water.xyz is not on the HPC node, embed it
            # inline the same way the pre-submit hook would. The
            # ``full_output`` key in the result carries the on-disk JSON
            # back to us (mace_mcp_hpc._mace_worker, lines 127-131) so
            # we can check converged without a follow-up transfer.
            from ase.io import read as ase_read

            from chemgraph.tools.ase_core import atoms_to_atomsdata
            from _smoke_utils import water_xyz_path

            atoms = ase_read(str(water_xyz_path()))
            inline = atoms_to_atomsdata(atoms).model_dump()

            job = {
                "input_structure_file": "ignored_by_inline_path",
                "output_result_file": "water_smoke_gc.json",
                "driver": "opt",
                "model": "medium-mpa-0",
                "device": os.environ.get("CG_SMOKE_DEVICE", "cuda"),
                "temperature": 298.15,
                "pressure": 101325.0,
                "fmax": 0.01,
                "steps": 100,
                "optimizer": "lbfgs",
                "inline_structure": inline,
            }
            fut = backend.submit(
                TaskSpec(
                    task_id="gc-mace-water-opt",
                    task_type="python",
                    callable=_mace_worker,
                    kwargs={"job": job},
                )
            )
            # First MACE run on the endpoint downloads the model + opt loop.
            mace_out = fut.result(timeout=6000)
            assert isinstance(mace_out, dict), type(mace_out)
            assert mace_out.get("status") == "success", f"opt failed: {mace_out}"
            energy = next(
                (mace_out[k] for k in ("single_point_energy", "energy", "final_energy") if k in mace_out),
                None,
            )
            assert energy is not None and energy < 0, f"bad energy: {mace_out}"

            full = mace_out.get("full_output") or {}
            if full:
                assert full.get("converged") is True, f"opt did not converge: {full.get('converged')!r}"
                assert full.get("success") is True, f"opt success=False: {full}"
                print(
                    f"       remote opt energy = {energy:.6f} eV  "
                    f"(converged={full.get('converged')}, wall={full.get('wall_time')}s)"
                )
            else:
                # full_output is attached by _mace_worker only when inline_structure
                # is set; we always pass inline above so this branch should not hit.
                print(
                    f"       remote opt energy = {energy:.6f} eV  "
                    "(WARNING: full_output not returned; convergence not verified)"
                )

    with r.check("backend.shutdown() is clean"):
        backend.shutdown()

    r.summary_and_exit()


if __name__ == "__main__":
    main()
