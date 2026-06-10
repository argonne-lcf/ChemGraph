"""Backend worker functions for gRASPA MCP tools.

This module intentionally contains no FastMCP/CGFastMCP objects or tool
decorators, keeping worker functions safe for Parsl/dill serialization.
"""

import os


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


def _ls_remote_files(path: str) -> list[str]:
    """Backend-side helper: list non-directory entries in *path*."""
    return sorted(
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    )
