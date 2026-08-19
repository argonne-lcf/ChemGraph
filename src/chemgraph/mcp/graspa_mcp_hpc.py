"""Backend-agnostic adsorption MCP server."""

import logging
import os
import re
from pathlib import Path

from chemgraph.execution.base import TaskSpec
from chemgraph.execution.config import get_transfer_manager
from chemgraph.execution.utils import resolve_structure_files
from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.mcp.transfer_tools import register_transfer_tools
from chemgraph.schemas.adsorption_schema import (
    AdsorptionEnsembleRequest,
    AdsorptionRequest,
)
from chemgraph.schemas.graspa_schema import graspa_input_schema_ensemble

logger = logging.getLogger(__name__)
_JOBS_FILE = Path("~/.chemgraph/graspa_jobs.json").expanduser()

mcp = CGFastMCP(
    name="ChemGraph Adsorption Tools",
    instructions="""
        Run pure-gas or mixture adsorption simulations with the engine
        selected in [adsorption]. Use run_adsorption_ensemble for new
        workflows and run_graspa_ensemble for legacy single-gas calls.
        Remote Globus Compute calls should provide remote_structure_files.
    """,
)


def _adsorption_worker(job: dict) -> dict:
    """Execute one fully resolved adsorption job on a backend worker."""

    from chemgraph.tools.adsorption_core import run_adsorption_core

    payload = dict(job)
    structure = payload.pop("_structure_name", None)
    runtime = payload.pop("_runtime")
    remote_file = payload.pop("remote_structure_file", None)
    if remote_file is not None:
        payload["input_structure_file"] = remote_file
    request = AdsorptionRequest.model_validate(payload)
    result = run_adsorption_core(request, runtime=runtime)
    return {"structure": structure, **result}


def _ls_remote_files(path: str) -> list[str]:
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if os.path.isfile(os.path.join(path, name))
        and name.lower().endswith(".cif")
    )


CGFastMCP._fix_module_for_pickle(_ls_remote_files)


def _condition_output(
    base_output: Path,
    structure_name: str,
    temperature: float,
    pressure: float,
    composition: str,
) -> str:
    safe_composition = re.sub(r"[^A-Za-z0-9_-]+", "_", composition)
    suffix = base_output.suffix or ".log"
    stem = base_output.stem
    name = (
        f"{structure_name}_T{temperature:g}_P{pressure:g}_"
        f"{safe_composition}_{stem}{suffix}"
    )
    return str(base_output.with_name(name))


def _remote_paths(params: AdsorptionEnsembleRequest) -> list[str] | None:
    if params.remote_structure_files:
        paths = sorted(params.remote_structure_files)
        invalid = [path for path in paths if not path.lower().endswith(".cif")]
        if invalid:
            raise ValueError(f"Remote adsorption inputs must be CIF files: {invalid}")
        return paths
    if not params.remote_structure_directory:
        return None

    mcp._ensure_backend()
    if not mcp._backend.shares_filesystem:
        raise ValueError(
            "remote_structure_directory requires a shared filesystem; "
            "use remote_structure_files with Globus Compute"
        )
    probe = TaskSpec(
        task_id="ls_remote_adsorption_dir",
        task_type="python",
        callable=_ls_remote_files,
        kwargs={"path": params.remote_structure_directory},
    )
    try:
        paths = mcp._backend.submit(probe).result(timeout=30)
    except Exception as exc:
        raise RuntimeError(
            f"Could not list {params.remote_structure_directory}: {exc}"
        ) from exc
    if not paths:
        raise ValueError(
            f"No CIF files found under {params.remote_structure_directory}"
        )
    return paths


def _expand_adsorption_ensemble(params: AdsorptionEnsembleRequest) -> list[dict]:
    from chemgraph.tools.adsorption_config import load_adsorption_runtime
    from chemgraph.tools.adsorption_drivers import get_adsorption_driver

    runtime = load_adsorption_runtime()
    driver = get_adsorption_driver(runtime.engine)
    base_output = Path(params.output_result_file)
    composition = "-".join(component.name for component in params.components)

    remote_paths = _remote_paths(params)
    if remote_paths is None:
        structure_paths, _ = resolve_structure_files(
            params.input_structures, extensions={".cif"}
        )
        sources = [(path.stem, str(path), False) for path in structure_paths]
    else:
        sources = [(Path(path).stem, path, True) for path in remote_paths]

    jobs = []
    for structure_name, structure_path, remote in sources:
        for condition in params.conditions:
            request = AdsorptionRequest(
                input_structure_file=structure_path,
                output_result_file=_condition_output(
                    base_output,
                    structure_name,
                    float(condition.temperature),
                    float(condition.pressure),
                    composition,
                ),
                temperature=condition.temperature,
                pressure=condition.pressure,
                n_cycles=params.n_cycles,
                cutoff=params.cutoff,
                components=params.components,
                engine_options=params.engine_options,
            )
            driver.validate_capabilities(request)
            job = request.model_dump(mode="json")
            job["_structure_name"] = structure_name
            job["_runtime"] = runtime.model_dump(mode="json")
            if remote:
                job.pop("input_structure_file")
                job["remote_structure_file"] = structure_path
            jobs.append(job)
    return jobs


@mcp.schema_fanout_tool(
    name="run_adsorption_ensemble",
    description="Run pure-gas or mixture adsorption over structures and conditions.",
    worker=_adsorption_worker,
    gpus_per_task=1,
)
def run_adsorption_ensemble(params: AdsorptionEnsembleRequest) -> list[dict]:
    return _expand_adsorption_ensemble(params)


@mcp.schema_fanout_tool(
    name="run_graspa_ensemble",
    description="Compatibility tool for single-component gRASPA ensembles.",
    worker=_adsorption_worker,
    gpus_per_task=1,
)
def run_graspa_ensemble(params: graspa_input_schema_ensemble) -> list[dict]:
    return _expand_adsorption_ensemble(params.to_adsorption_request())


_transfer_manager = get_transfer_manager()
if _transfer_manager is not None:
    register_transfer_tools(mcp, _transfer_manager)
    logger.info("Registered Globus Transfer tools on adsorption MCP server.")


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    mcp.init_backend(tracker_kwargs={"persist_file": _JOBS_FILE})
    try:
        run_mcp_server(mcp, default_port=9001)
    finally:
        mcp.shutdown_backend()
