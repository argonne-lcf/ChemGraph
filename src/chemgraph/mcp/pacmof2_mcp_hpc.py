"""Backend-agnostic PACMOF2 MCP server.

Uses :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`. Tool functions are
plain computation -- the framework handles backend submission
(local / parsl / ensemble_launcher / globus_compute), future resolution,
and async job tracking.

PACMOF2 assigns ML partial atomic charges to MOF CIFs. The ensemble
expander emits one job per structure and supports both local input
directories and pre-staged remote directories (mirrors the gRASPA
server's local/remote modes). PACMOF2 is CPU-only and fast, so the
default TaskSpec resource hints (single node, no GPU) are used.

Nothing requiring the backend is initialised at import time so worker
subprocesses (EnsembleLauncher, Globus Compute) can re-import this
module safely.
"""

import logging
import os
from pathlib import Path

from chemgraph.execution.base import TaskSpec
from chemgraph.execution.config import get_transfer_manager
from chemgraph.execution.utils import resolve_structure_files
from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.mcp.transfer_tools import register_transfer_tools
from chemgraph.schemas.pacmof2_schema import pacmof2_input_schema_ensemble

logger = logging.getLogger(__name__)

_JOBS_FILE = Path("~/.chemgraph/pacmof2_jobs.json").expanduser()

mcp = CGFastMCP(
    name="ChemGraph PACMOF2 Tools",
    instructions="""
        You expose tools for assigning PACMOF2 machine-learned partial
        atomic charges to MOF structures (CIF files). The available tools
        are:
        1. run_pacmof2_ensemble: assign charges to every CIF in a
           directory (or list). Local mode uses input_structures; remote
           mode uses remote_structure_directory (pre-stage files first
           with transfer_files).
        2. check_job_status / get_job_results / list_jobs / cancel_job:
           HPC job batch management. Job state persists across sessions.
        3. transfer_files / check_transfer_status / list_remote_files
           (when Globus Transfer is configured): stage input files on the
           remote HPC filesystem before running ensembles in remote mode.

        Guidelines:
        - Use each tool only when its input schema matches the user
          request.
        - PACMOF2 writes a charged CIF ('{stem}{identifier}.cif') next to
          each input CIF, with an _atom_site_charge column. In remote mode
          the output lands on the endpoint's filesystem; pull it back with
          a transfer tool if needed.
        - net_charge is 0 for neutral MOFs, an int/float for a charged
          framework, or a dict for ionic MOFs.
        - Do not guess numerical values; report tool errors exactly as
          they occur.
        - Keep responses compact -- full per-atom charges live in the
          output CIF; tools return only a summary.
        - When returning paths, use absolute paths.
        - When a tool returns status='submitted' with a batch_id, use
          check_job_status to poll for progress before calling
          get_job_results. Job state is persisted across sessions.
    """,
)


# ── Worker (runs on the backend) ───────────────────────────────────────


def _pacmof2_worker(job: dict) -> dict:
    """Assign PACMOF2 charges to a single structure on a backend worker."""
    from chemgraph.schemas.pacmof2_schema import pacmof2_input_schema
    from chemgraph.tools.pacmof2_tools import run_pacmof2_core

    job = dict(job)
    structure = job.pop("_structure_name", None)

    remote_file = job.pop("remote_structure_file", None)
    if remote_file is not None:
        job["input_structure_file"] = remote_file

    params = pacmof2_input_schema(**job)
    result = run_pacmof2_core(params)

    if isinstance(result, dict):
        merged = {"structure": structure, **result}
        merged.setdefault("status", "success")
        return merged
    return {"structure": structure, "result": result, "status": "success"}


# ── Remote directory listing (runs on the backend) ─────────────────────


def _ls_remote_files(path: str) -> list[str]:
    """Backend-side helper: list non-directory entries in *path*."""
    return sorted(
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    )


# Submitted as a bare ``callable=`` TaskSpec (not via a decorator), so it must
# be fixed explicitly for pickle-by-reference when run as ``__main__``. Mirrors
# the equivalent fix in graspa_mcp_hpc.py.
CGFastMCP._fix_module_for_pickle(_ls_remote_files)


# ── Ensemble fanout ────────────────────────────────────────────────────


def _expand_pacmof2_ensemble(params: pacmof2_input_schema_ensemble) -> list[dict]:
    """Server-side expansion of an ensemble request into per-job dicts.

    Local mode: enumerates ``input_structures`` on this host.
    Remote mode: submits a one-shot probe task to the backend to list
    files under ``remote_structure_directory``, then builds per-file jobs
    that the worker reads directly from the remote filesystem.
    """
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

        file_names = [f for f in file_names if f.lower().endswith(".cif")]
        if not file_names:
            raise ValueError(
                f"No CIF files found under remote directory {remote_dir}."
            )

        return [
            {
                "_structure_name": Path(fname).stem,
                "remote_structure_file": f"{remote_dir}/{fname}",
                "identifier": params.identifier,
                "adjust_charge_method": params.adjust_charge_method,
                "net_charge": params.net_charge,
            }
            for fname in file_names
        ]

    if not params.input_structures:
        raise ValueError(
            "Either input_structures or remote_structure_directory "
            "must be provided."
        )

    structure_files, _ = resolve_structure_files(
        params.input_structures, extensions={".cif"}
    )
    return [
        {
            "_structure_name": struct_path.stem,
            "input_structure_file": str(struct_path),
            "identifier": params.identifier,
            "adjust_charge_method": params.adjust_charge_method,
            "net_charge": params.net_charge,
        }
        for struct_path in structure_files
    ]


@mcp.schema_fanout_tool(
    name="run_pacmof2_ensemble",
    description=(
        "Assign PACMOF2 ML partial atomic charges to every CIF in a "
        "directory (or list). Local mode uses input_structures; remote "
        "mode uses remote_structure_directory (pre-stage files first with "
        "transfer_files)."
    ),
    worker=_pacmof2_worker,
)
def run_pacmof2_ensemble(params: pacmof2_input_schema_ensemble) -> list[dict]:
    return _expand_pacmof2_ensemble(params)


# ── Globus Transfer (registered only when configured) ──────────────────

_transfer_manager = get_transfer_manager()
if _transfer_manager is not None:
    register_transfer_tools(mcp, _transfer_manager)
    logger.info("Registered Globus Transfer tools on PACMOF2 MCP server.")


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})

    try:
        run_mcp_server(mcp, default_port=9009)
    finally:
        mcp.shutdown_backend()
