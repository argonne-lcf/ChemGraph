"""Deprecated Parsl MCP entry point; retained for existing client configurations."""

import asyncio
from concurrent.futures import Future
import os
from pathlib import Path
import tempfile
import uuid
import warnings

from mcp.server.fastmcp import FastMCP

from chemgraph.execution.base import TaskSpec
from chemgraph.execution.utils import gather_futures, write_results_jsonl
from chemgraph.mcp.cg_fastmcp import CGFastMCP
from chemgraph.mcp.graspa_mcp_hpc import (
    _expand_graspa_ensemble, _graspa_worker, _job_metadata,
)
from chemgraph.schemas.graspa_schema import graspa_input_schema_ensemble

warnings.warn(
    "chemgraph.mcp.graspa_mcp_parsl is deprecated; use chemgraph.mcp.graspa_mcp_hpc "
    "with CHEMGRAPH_EXECUTION_BACKEND=parsl.", DeprecationWarning, stacklevel=2,
)

_backend = None
mcp = FastMCP(
    name="ChemGraph Graspa Tools",
    instructions="""run_graspa_ensemble runs single-component H2O, CO2, or N2 gRASPA-SYCL and returns a JSONL
    summary path. Temperature is in K, pressure in Pa, uptake in mol/kg. One
    structure and one condition run a single simulation. Report failures and
    use actual returned artifact paths. Migrate to graspa_mcp_hpc for job tools.""",
)


def _get_backend():
    global _backend
    if _backend is None:
        from chemgraph.execution.config import get_backend

        _backend = get_backend(
            backend_name="parsl", system=os.getenv("COMPUTE_SYSTEM", "polaris"),
        )
    return _backend


def run_graspa_parsl_app(job: dict) -> Future:
    """Retain the historical future-returning callable with lazy Parsl setup."""
    try:
        if hasattr(job, "model_dump"):
            job = job.model_dump()
        if not isinstance(job, dict):
            raise TypeError("run_graspa_parsl_app expects a dict or gRASPA input model")
        job = dict(job)
        job.setdefault("_job_id", uuid.uuid4().hex)
        return _get_backend().submit(TaskSpec(
            task_id=job["_job_id"], task_type="python",
            callable=_graspa_worker, kwargs={"job": job},
        ))
    except Exception as exc:
        future = Future()
        future.set_exception(exc)
        return future


def _write_summary(request: dict, results: list[dict]) -> str:
    """Write the legacy summary with output roots resolved on the worker."""
    from chemgraph.tools.graspa_core import resolve_output_directory

    root = resolve_output_directory(graspa_input_schema_ensemble(**request))
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="ensemble-", dir=root))
    path = directory / "simulation_results.jsonl"
    successful, total = write_results_jsonl(results, path, append=False)
    return (
        f"Ensemble execution completed. Ran {total} tasks "
        f"({successful} successful). Detailed results saved to '{path}'."
    )


CGFastMCP._fix_module_for_pickle(_write_summary)


@mcp.tool(name="run_graspa_ensemble")
async def run_graspa_ensemble(params: graspa_input_schema_ensemble):
    """Run a single-adsorbate ensemble and return an isolated JSONL summary path."""
    jobs = await _expand_graspa_ensemble(params, backend=_get_backend())
    pending = [(_job_metadata(job), run_graspa_parsl_app(job)) for job in jobs]
    results = await gather_futures(pending)
    future = _get_backend().submit(TaskSpec(
        task_id=f"graspa_summary_{uuid.uuid4().hex}", task_type="python",
        callable=_write_summary,
        kwargs={"request": params.model_dump(), "results": results},
    ))
    return await asyncio.wrap_future(future)


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server
    try:
        run_mcp_server(mcp, default_port=9001)
    finally:
        if _backend is not None:
            _backend.shutdown()
