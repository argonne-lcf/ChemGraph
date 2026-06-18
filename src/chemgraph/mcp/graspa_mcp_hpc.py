"""Backend-agnostic gRASPA MCP server.

Uses :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`. Tool functions are
plain computation -- the framework handles backend submission, future
resolution, and async job tracking.

The ensemble expander emits one job per ``(structure, condition)`` pair
and supports both local input directories and pre-staged remote
directories (mirrors the MACE server's local/remote modes).

Nothing requiring the backend is initialised at import time so worker
subprocesses (EnsembleLauncher, Globus Compute) can re-import this
module safely.
"""

import logging
import os
from pathlib import Path

from chemgraph.execution.base import TaskSpec
from chemgraph.execution.config import get_transfer_manager
from chemgraph.execution.utils import (
    make_per_structure_output,
    resolve_structure_files,
)
from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.mcp.transfer_tools import register_transfer_tools
from chemgraph.schemas.graspa_schema import graspa_input_schema_ensemble

logger = logging.getLogger(__name__)

_JOBS_FILE = Path("~/.chemgraph/graspa_jobs.json").expanduser()

mcp = CGFastMCP(
    name="ChemGraph Graspa Tools",
    instructions="""
        You expose tools for running gRASPA simulations and reading
        their results.  The available tools are:
        1. run_graspa_ensemble: run gRASPA calculations over every
           structure in a directory at one or more (T, P) conditions.
           Local mode uses input_structures; remote mode uses
           remote_structure_directory (pre-stage files first with
           transfer_files).
        2. check_job_status / get_job_results / list_jobs / cancel_job:
           HPC job batch management. Job state persists across sessions.
        3. transfer_files / check_transfer_status / list_remote_files
           (when Globus Transfer is configured): stage input files on
           the remote HPC filesystem before running ensembles in remote
           mode.

        Guidelines:
        - Use each tool only when its input schema matches the user
          request.
        - Do not guess numerical values; report tool errors exactly as
          they occur.
        - Keep responses compact -- full results are written to the
          output files defined in the schemas.
        - When returning paths, use absolute paths.
        - Energies are in eV and wall times are in seconds.
        - When a tool returns status='submitted' with a batch_id, use
          check_job_status to poll for progress before calling
          get_job_results.  Job state is persisted across sessions.
    """,
)


# ── Worker (runs on the backend) ───────────────────────────────────────


def _graspa_worker(job: dict) -> dict:
    """Execute a single gRASPA simulation on a backend worker."""
    from chemgraph.schemas.graspa_schema import graspa_input_schema
    from chemgraph.tools.graspa_tools import run_graspa_core

    job = dict(job)
    structure = job.pop("_structure_name", None)
    temperature = job.get("temperature")
    pressure = job.get("pressure")

    remote_file = job.pop("remote_structure_file", None)
    if remote_file is not None:
        job["input_structure_file"] = remote_file
        if not os.path.isabs(job.get("output_result_file", "")):
            job["output_result_file"] = os.path.join(
                os.path.dirname(remote_file),
                job.get("output_result_file", "raspa.log"),
            )

    params = graspa_input_schema(**job)
    result = run_graspa_core(params)

    if isinstance(result, dict):
        merged = {
            "structure": structure,
            "temperature": temperature,
            "pressure": pressure,
            **result,
        }
        merged.setdefault("status", "success")
        return merged
    return {
        "structure": structure,
        "temperature": temperature,
        "pressure": pressure,
        "result": result,
        "status": "success",
    }


# Note: ``_graspa_worker`` is registered via ``@mcp.schema_fanout_tool`` below,
# which fixes its module for pickling automatically; no explicit fix is needed.


# ── Ensemble fanout ────────────────────────────────────────────────────


def _ls_remote_files(path: str) -> list[str]:
    """Backend-side helper: list non-directory entries in *path*."""
    return sorted(
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    )


# Submitted as a bare ``callable=`` TaskSpec (not via a decorator), so it must
# be fixed explicitly for pickle-by-reference when run as ``__main__``. Mirrors
# the equivalent fix in mace_mcp_hpc.py.
CGFastMCP._fix_module_for_pickle(_ls_remote_files)


def _expand_graspa_ensemble(params: graspa_input_schema_ensemble) -> list[dict]:
    """Server-side expansion of an ensemble request into per-job dicts.

    Local mode: enumerates ``input_structures`` on this host.
    Remote mode: submits a one-shot probe task to the backend to list
    files under ``remote_structure_directory``, then builds per-file
    jobs that the worker reads directly from the remote filesystem.
    """
    base_output = Path(params.output_result_file)

    if params.remote_structure_directory:
        remote_dir = params.remote_structure_directory
        mcp._ensure_backend()
        probe = TaskSpec(
            task_id="ls_remote_dir",
            task_type="python",
            callable=_ls_remote_files,
            kwargs={"path": remote_dir},
        )
        fut = mcp._backend.submit(probe)
        try:
            file_names = fut.result(timeout=30)
        except Exception as exc:
            raise RuntimeError(
                f"Could not list remote directory {remote_dir}: {exc}"
            ) from exc

        # Filter to CIF files (gRASPA expects CIFs).
        file_names = [f for f in file_names if f.lower().endswith(".cif")]
        if not file_names:
            raise ValueError(
                f"No CIF files found under remote directory {remote_dir}."
            )

        jobs = []
        for fname in file_names:
            mof_name = Path(fname).stem
            for condition in params.conditions:
                per_output = make_per_structure_output(Path(fname), base_output)
                jobs.append(
                    {
                        "_structure_name": mof_name,
                        "remote_structure_file": f"{remote_dir}/{fname}",
                        "output_result_file": str(per_output),
                        "temperature": condition.temperature,
                        "pressure": condition.pressure,
                        "adsorbate": params.adsorbate,
                        "n_cycles": params.n_cycles,
                    }
                )
        return jobs

    if not params.input_structures:
        raise ValueError(
            "Either input_structures or remote_structure_directory "
            "must be provided."
        )

    structure_files, _ = resolve_structure_files(
        params.input_structures, extensions={".cif"}
    )
    jobs = []
    for struct_path in structure_files:
        mof_name = struct_path.stem
        for condition in params.conditions:
            per_output = make_per_structure_output(struct_path, base_output)
            jobs.append(
                {
                    "_structure_name": mof_name,
                    "input_structure_file": str(struct_path),
                    "output_result_file": str(per_output),
                    "temperature": condition.temperature,
                    "pressure": condition.pressure,
                    "adsorbate": params.adsorbate,
                    "n_cycles": params.n_cycles,
                }
            )
    return jobs


@mcp.schema_fanout_tool(
    name="run_graspa_ensemble",
    description=(
        "Run gRASPA calculations over every structure in a directory at "
        "one or more (temperature, pressure) conditions. Local mode "
        "uses input_structures; remote mode uses "
        "remote_structure_directory (pre-stage files first with "
        "transfer_files)."
    ),
    worker=_graspa_worker,
)
def run_graspa_ensemble(params: graspa_input_schema_ensemble) -> list[dict]:
    return _expand_graspa_ensemble(params)


# ── Globus Transfer (registered only when configured) ──────────────────

_transfer_manager = get_transfer_manager()
if _transfer_manager is not None:
    register_transfer_tools(mcp, _transfer_manager)
    logger.info("Registered Globus Transfer tools on gRASPA MCP server.")


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})

    try:
        run_mcp_server(mcp, default_port=9001)
    finally:
        mcp.shutdown_backend()
