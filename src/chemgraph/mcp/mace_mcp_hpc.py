"""Backend-agnostic MACE MCP server.

Uses :class:`~chemgraph.mcp.cg_fastmcp.CGFastMCP`. Tool functions are
plain computation -- the framework handles backend submission, future
resolution, and async job tracking.

Transport (local-file embedding, pre-staged remote-path passthrough)
lives in a single pre-submit hook so the tool bodies stay simple. The
hook rewrites :class:`~chemgraph.execution.base.TaskSpec` instances
before submission to attach an inline structure when the input file
exists on the submitting host, leaving the path untouched when it
does not (assumed to be remote).

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
from chemgraph.schemas.mace_parsl_schema import (
    mace_input_schema,
    mace_input_schema_ensemble,
)
from chemgraph.tools.parsl_tools import extract_output_json, run_mace_core

logger = logging.getLogger(__name__)

_JOBS_FILE = Path("~/.chemgraph/mace_jobs.json").expanduser()
_MACE_MP_ALIASES = {"mace_mp", "mace-mp", "MACE-MP", "mace_MP"}

mcp = CGFastMCP(
    name="ChemGraph MACE Tools",
    instructions="""
        You expose tools for running MACE simulations and reading their results.
        The available tools are:
        1. run_mace_single: run a single MACE calculation.
        2. run_mace_ensemble: run MACE calculations over every structure in a
           directory (local or pre-staged remote).
        3. extract_output_json: load simulation results from a JSON file.
        4. check_job_status / get_job_results / list_jobs / cancel_job: HPC
           job batch management. Job state persists across sessions.
        5. transfer_files / check_transfer_status / list_remote_files
           (when Globus Transfer is configured): stage input files on the
           remote HPC filesystem before running ensembles in remote mode.

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they
          occur.
        - Keep responses compact -- full results are written to the output
          files defined in the schemas.
        - When returning paths, use absolute paths.
        - Energies are in eV and wall times are in seconds.
        - When a tool returns status='submitted' with a batch_id, call
          get_job_results(batch_id) to retrieve results. If still pending,
          report the batch_id so the user can check later -- job state is
          persisted across sessions.
        - For the `model` field, pass a MACE foundation model name (e.g.
          'medium-mpa-0'). 'mace_mp' is the calculator type, not a model
          name -- do not pass it.
    """,
)


# ── Worker (runs on the backend) ───────────────────────────────────────


def _mace_worker(job: dict) -> dict:
    """Execute a single MACE simulation on a backend worker.

    Accepts a *job dict* (not the schema) so the pre-submit hook can
    attach transport keys ``inline_structure`` / ``remote_structure_file``
    before submission.
    """
    import tempfile

    job = dict(job)

    # Pre-staged remote file: use the path directly on the worker FS.
    remote_file = job.pop("remote_structure_file", None)
    if remote_file is not None:
        job["input_structure_file"] = remote_file
        if not os.path.isabs(job.get("output_result_file", "")):
            job["output_result_file"] = os.path.join(
                os.path.dirname(remote_file),
                job.get("output_result_file", "output.json"),
            )

    # Inline structure: materialise on the worker's filesystem.
    inline = job.pop("inline_structure", None)
    if inline is not None:
        from ase import Atoms
        from ase.io import write as ase_write

        atoms = Atoms(
            numbers=inline["numbers"],
            positions=inline["positions"],
            cell=inline.get("cell"),
            pbc=inline.get("pbc"),
        )
        tmpdir = tempfile.mkdtemp(prefix="chemgraph_mace_")
        xyz_path = os.path.join(tmpdir, "structure.xyz")
        ase_write(xyz_path, atoms)
        job["input_structure_file"] = xyz_path
        if not os.path.isabs(job.get("output_result_file", "")):
            job["output_result_file"] = os.path.join(
                tmpdir, job.get("output_result_file", "output.json")
            )

    params = mace_input_schema(**job)
    result = run_mace_core(params)
    return result


# Force pickle-by-reference for callables that the transport hook installs
# as `task.callable`. Without this, dill sees `__module__ == "__main__"`
# (this file is run as ``python -m chemgraph.mcp.mace_mcp_hpc``) and falls
# back to pickle-by-value, which walks the module's globals and tries to
# serialize the dynamic ``run_mace_singleArguments`` class held by
# ``mcp._tool_manager._tools[...].fn_metadata.arg_model`` -- that class
# was created by ``pydantic.create_model`` with a ``__module__`` it was
# never registered into, so dill raises a PicklingError.
CGFastMCP._fix_module_for_pickle(_mace_worker)


# ── Pre-submit transport hook ──────────────────────────────────────────


def _embed_inline_if_local(job: dict) -> None:
    """Mutate *job* in-place: attach inline_structure when the input
    file is readable on the submitting host (and no other transport
    key has already been set)."""
    if job.get("remote_structure_file") or job.get("inline_structure"):
        return
    input_file = job.get("input_structure_file")
    if not input_file or not os.path.isfile(input_file):
        return  # remote path -- worker will read it directly

    from ase.io import read as ase_read

    from chemgraph.tools.ase_core import atoms_to_atomsdata

    atoms = ase_read(input_file)
    job["inline_structure"] = atoms_to_atomsdata(atoms).model_dump()


def _normalize_model(job: dict) -> None:
    """Map calculator-type aliases to a valid foundation model name."""
    if job.get("model") in _MACE_MP_ALIASES:
        job["model"] = "medium-mpa-0"


def _backend_shares_fs() -> bool:
    """Whether the active backend shares the server's filesystem.

    When it does, inline embedding (and the worker's ``/tmp`` round-trip)
    is unnecessary -- the worker reads ``input_structure_file`` directly.
    Defaults to ``True`` (skip embedding) when no backend exists yet."""
    backend = getattr(mcp, "_backend", None)
    return getattr(backend, "shares_filesystem", True)


def _mace_transport_hook(task: TaskSpec) -> TaskSpec:
    """Route single-tool calls to the dict-based worker and embed
    local structures only when the backend has no shared filesystem."""
    logger.debug(
        "mace transport hook: task_id=%s callable=%s",
        task.task_id,
        getattr(task.callable, "__qualname__", task.callable),
    )
    if task.callable is run_mace_single:
        params = task.kwargs.get("params")
        if params is None:
            return task
        job = (
            params.model_dump() if hasattr(params, "model_dump") else dict(params)
        )
        _normalize_model(job)
        if not _backend_shares_fs():
            _embed_inline_if_local(job)
        task.callable = _mace_worker
        task.kwargs = {"job": job}
    elif task.callable is _mace_worker:
        job = dict(task.kwargs.get("job", {}))
        _normalize_model(job)
        if not _backend_shares_fs():
            _embed_inline_if_local(job)
        task.kwargs = {"job": job}
    return task


mcp.set_pre_submit_hook(_mace_transport_hook)


# ── Single-structure tool ──────────────────────────────────────────────


@mcp.tool(
    name="run_mace_single",
    description="Run a single MACE calculation",
)
def run_mace_single(params: mace_input_schema) -> dict:
    """Run a single MACE calculation on the configured backend.

    The pre-submit hook rewrites this call to invoke ``_mace_worker``
    on the backend with a job dict that may carry an embedded inline
    structure (when the input file exists locally) or a remote path
    (when it does not).
    """
    # Direct-call fallback path (no hook registered) -- normalises and
    # delegates to the same worker.
    job = params.model_dump()
    _normalize_model(job)
    return _mace_worker(job)


# ── Ensemble fanout ────────────────────────────────────────────────────


def _ls_remote_files(path: str) -> list[str]:
    """Backend-side helper: list non-directory entries in *path*."""
    return sorted(
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    )


CGFastMCP._fix_module_for_pickle(_ls_remote_files)


def _expand_mace_ensemble(params: mace_input_schema_ensemble) -> list[dict]:
    """Server-side expansion of an ensemble request into per-file jobs.

    Local mode: enumerates ``input_structure_directory`` on this host.
    Remote mode: submits a one-shot probe task to the backend to list
    files under ``remote_structure_directory``, then builds per-file
    jobs that the worker reads directly from the remote filesystem.
    """
    shared = {
        "output_result_file": params.output_result_file,
        "driver": params.driver,
        "model": params.model,
        "device": params.device,
        "temperature": params.temperature,
        "pressure": params.pressure,
        "fmax": params.fmax,
        "steps": params.steps,
        "optimizer": params.optimizer,
    }
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

        jobs = []
        for fname in file_names:
            per_output = make_per_structure_output(Path(fname), base_output)
            job = {**shared}
            job["remote_structure_file"] = f"{remote_dir}/{fname}"
            job["output_result_file"] = str(per_output)
            jobs.append(job)
        return jobs

    if not params.input_structure_directory:
        raise ValueError(
            "Either input_structure_directory or remote_structure_directory "
            "must be provided."
        )

    structure_files, _ = resolve_structure_files(params.input_structure_directory)
    return [
        {
            **shared,
            "input_structure_file": str(f),
            "output_result_file": str(make_per_structure_output(f, base_output)),
        }
        for f in structure_files
    ]


@mcp.schema_fanout_tool(
    name="run_mace_ensemble",
    description=(
        "Run MACE calculations over every structure in a directory. "
        "Local mode uses input_structure_directory; remote mode uses "
        "remote_structure_directory (pre-stage files first with "
        "transfer_files)."
    ),
    worker=_mace_worker,
)
def run_mace_ensemble(params: mace_input_schema_ensemble) -> list[dict]:
    return _expand_mace_ensemble(params)


# ── Orchestration tools (no backend involvement) ───────────────────────


mcp.add_tool(
    extract_output_json,
    name="extract_output_json",
    description="Load simulation results from an output JSON file.",
)


# ── Globus Transfer (registered only when configured) ──────────────────

_transfer_manager = get_transfer_manager()
if _transfer_manager is not None:
    register_transfer_tools(mcp, _transfer_manager)
    logger.info("Registered Globus Transfer tools on MACE MCP server.")


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})

    try:
        run_mcp_server(mcp, default_port=9004)
    finally:
        mcp.shutdown_backend()
