"""Backend-agnostic, lazily configured H2O gRASPA-SYCL MCP server."""

import asyncio
import math
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
import uuid

from chemgraph.execution.base import TaskSpec
from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.schemas.graspa_schema import graspa_input_schema_ensemble

_JOBS_FILE = Path("~/.chemgraph/graspa_jobs.json").expanduser()
mcp = CGFastMCP(
    name="ChemGraph Graspa Tools",
    instructions="""
        run_graspa_ensemble runs H2O adsorption with gRASPA-SYCL. One structure
        and one condition also run a single simulation. Temperature is in K,
        pressure in Pa, uptake in mol/kg, and wall_time in seconds.
        input_structures requires a shared filesystem. For remote backends,
        pre-stage CIFs using transfer_files when configured, then provide
        remote_structure_directory. Output roots and executable configuration
        belong to the worker, not the client host. Each run has a unique directory.
        Return actual artifact paths from results; never construct them yourself.
        If status is submitted, retain batch_id, poll check_job_status, then call
        get_job_results. list_jobs and cancel_job manage tracked batches;
        cancellation is best effort for pending tasks, not running processes.
        A completed batch can contain failed simulations. Report their errors;
        never interpret null uptake as zero or a pending batch as complete.
        raspa.log is plain text, not JSONL. Do not invent or round numerical data.
    """,
)


def _job_metadata(job: dict) -> dict:
    """Identify a simulation even when it fails before entering the worker."""
    source = job.get("remote_structure_file", job.get("input_structure_file"))
    return {
        "task_id": job.get("_job_id"),
        "job_id": job.get("_job_id"),
        "structure": job.get("_structure_name") or (Path(source).stem if source else None),
        "input_structure_file": source,
        "temperature": job.get("temperature"),
        "pressure": job.get("pressure"),
        "temperature_in_K": job.get("temperature"),
        "pressure_in_Pa": job.get("pressure"),
        "adsorbate": job.get("adsorbate"),
        "uptake_in_mol_kg": None,
    }


def _graspa_worker(job: dict) -> dict:
    """Run on the backend and normalize preparation and execution failures."""
    metadata = _job_metadata(job)
    artifacts = {}
    try:
        from chemgraph.schemas.graspa_schema import graspa_input_schema
        from chemgraph.tools.graspa_core import run_graspa_core

        values = {key: value for key, value in job.items() if not key.startswith("_")}
        remote = values.pop("remote_structure_file", None)
        if remote is not None:
            values["input_structure_file"] = remote
        params = graspa_input_schema(**values)
        metadata = _job_metadata({**params.model_dump(), **job})
        result = run_graspa_core(params)
        if isinstance(result, dict):
            artifacts = {
                key: result[key]
                for key in (
                    "run_id", "run_dir", "stdout_path", "stderr_path",
                    "results_path", "cif_path",
                )
                if key in result
            }
        if not isinstance(result, dict) or result.get("status") not in {"success", "failure"}:
            raise ValueError("gRASPA worker returned an invalid result")
        if result["status"] == "success":
            uptake = result.get("uptake_in_mol_kg")
            if (
                isinstance(uptake, bool)
                or not isinstance(uptake, (int, float))
                or not math.isfinite(uptake)
                or uptake < 0
            ):
                raise ValueError("gRASPA worker returned invalid uptake")
        else:
            result = {**result, "uptake_in_mol_kg": None}
        # Request identity must survive even a malformed core response.
        identity = {key: value for key, value in metadata.items() if key != "uptake_in_mol_kg"}
        return {**result, **identity}
    except Exception as exc:
        return {
            **artifacts, **metadata, "status": "failure",
            "error_type": type(exc).__name__, "message": str(exc),
        }


def _ls_remote_files(path: str) -> list[str]:
    """Discover CIFs and resolve their absolute paths on the executing host."""
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Not a CIF directory: {directory}")
    return [
        str(p.resolve()) for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() == ".cif"
    ]


CGFastMCP._fix_module_for_pickle(_ls_remote_files)


def _remote_cif_path(path: str) -> PurePath:
    """Validate worker paths without applying the MCP host's OS rules."""
    if isinstance(path, str):
        parsed = PureWindowsPath(path)
        if not parsed.is_absolute():
            parsed = PurePosixPath(path)
        if parsed.is_absolute() and parsed.suffix.lower() == ".cif":
            return parsed
    raise ValueError("Discovery must return absolute CIF paths")


def _local_structure_files(source: str | list[str]) -> list[str]:
    from chemgraph.tools.ase_core import _resolve_existing_path

    if isinstance(source, list):
        paths = [
            Path(_resolve_existing_path(str(Path(p).expanduser()))).resolve()
            for p in source
        ]
    else:
        directory = Path(source).expanduser().resolve()
        if not directory.is_dir():
            raise ValueError(f"Not a CIF directory: {directory}")
        paths = [
            p.resolve() for p in sorted(directory.iterdir())
            if p.is_file() and p.suffix.lower() == ".cif"
        ]
    if not paths:
        raise ValueError("No CIF files found to simulate")
    invalid = [str(p) for p in paths if not p.is_file() or p.suffix.lower() != ".cif"]
    if invalid:
        raise ValueError(f"Inputs must be existing CIF files: {invalid}")
    return [str(p) for p in paths]


async def _expand_graspa_ensemble(
    params: graspa_input_schema_ensemble, backend=None,
) -> list[dict]:
    """Expand a validated request without blocking MCP's event loop."""
    params = graspa_input_schema_ensemble.model_validate(params.model_dump())
    if backend is None:
        mcp._ensure_backend()
        backend = mcp._backend
    remote = params.remote_structure_directory is not None
    if remote:
        probe = TaskSpec(
            task_id=f"graspa_discovery_{uuid.uuid4().hex}", task_type="python",
            callable=_ls_remote_files, kwargs={"path": params.remote_structure_directory},
        )
        try:
            future = backend.submit(probe)
            paths = await asyncio.wait_for(
                asyncio.wrap_future(future), params.discovery_timeout_seconds,
            )
            if not paths:
                raise ValueError("No CIF files found")
            if not isinstance(paths, list):
                raise ValueError("Discovery must return absolute CIF paths")
            names = [_remote_cif_path(path).stem for path in paths]
        except Exception as exc:
            raise RuntimeError(
                f"Could not discover CIFs in {params.remote_structure_directory}: "
                f"{type(exc).__name__}: {exc}. Discovery includes queue time; "
                "check the staged directory and backend, or adjust "
                "discovery_timeout_seconds. No simulations were submitted."
            ) from exc
    else:
        if not backend.shares_filesystem:
            raise ValueError(
                "input_structures requires a shared filesystem; pre-stage CIFs "
                "and use remote_structure_directory"
            )
        paths = _local_structure_files(params.input_structures)
        names = [Path(path).stem for path in paths]
    options = params.model_dump(exclude={
        "input_structures", "remote_structure_directory",
        "conditions", "discovery_timeout_seconds",
    })
    input_key = "remote_structure_file" if remote else "input_structure_file"
    return [
        {**options, **condition.model_dump(), input_key: path,
         "_structure_name": name, "_job_id": uuid.uuid4().hex}
        for path, name in zip(paths, names) for condition in params.conditions
    ]


@mcp.schema_fanout_tool(
    name="run_graspa_ensemble", worker=_graspa_worker, metadata=_job_metadata,
    description="Run H2O gRASPA-SYCL for every CIF/condition pair using local shared files or a pre-staged remote directory.",
)
async def run_graspa_ensemble(params: graspa_input_schema_ensemble) -> list[dict]:
    return await _expand_graspa_ensemble(params)


def main():
    """Configure job/transfer tools only when explicitly starting the server."""
    from chemgraph.execution.config import get_transfer_manager
    from chemgraph.mcp.server_utils import run_mcp_server
    from chemgraph.mcp.transfer_tools import register_transfer_tools

    mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})
    try:
        transfer_manager = get_transfer_manager()
        if transfer_manager is not None:
            register_transfer_tools(mcp, transfer_manager)
        run_mcp_server(mcp, default_port=9001)
    finally:
        mcp.shutdown_backend()


if __name__ == "__main__":
    main()
