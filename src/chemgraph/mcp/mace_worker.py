"""Backend worker functions for MACE MCP tools.

This module intentionally contains no FastMCP/CGFastMCP objects or tool
decorators. Parsl/dill serializes worker functions by walking their module
globals, so backend workers must live outside modules that contain FastMCP's
runtime-generated argument classes.

The public ``_mace_worker`` shells the actual MACE call into a fresh Python
subprocess. Running MACE directly inside a Parsl worker on Aurora hangs
the worker indefinitely (no Python exception, no OS-level kill signal) —
the failure is silent and only happens inside Parsl's
``process_worker_pool.py`` process model. A clean interpreter dodges it,
so we pay the per-call subprocess cost (~3-5s of Python startup +
``module load frameworks`` env) to keep MACE working.
"""

import json
import os
import subprocess
import sys
import tempfile

_SUBPROCESS_TIMEOUT_S = 3600
_SUBRUNNER_MODULE = "chemgraph.mcp._mace_subrunner"


def _mace_worker(job: dict) -> dict:
    """Run one MACE simulation in an isolated subprocess.

    Materializes *job* to a temp JSON file and invokes
    :mod:`chemgraph.mcp._mace_subrunner` in a child interpreter. The child
    writes the result back to a sibling JSON file which we read and return.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".job.json", delete=False, encoding="utf-8",
    ) as job_fh:
        json.dump(job, job_fh)
        job_path = job_fh.name
    result_path = f"{job_path}.result.json"

    try:
        completed = subprocess.run(
            [sys.executable, "-m", _SUBRUNNER_MODULE, job_path, result_path],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
        )
        if completed.returncode != 0:
            return {
                "status": "failure",
                "error_type": "SubprocessError",
                "message": (
                    f"MACE subprocess exited with {completed.returncode}. "
                    f"stderr tail: {completed.stderr[-500:]}"
                ),
            }
        if not os.path.isfile(result_path):
            return {
                "status": "failure",
                "error_type": "SubprocessError",
                "message": (
                    "MACE subprocess exited 0 but wrote no result file. "
                    f"stdout tail: {completed.stdout[-500:]}"
                ),
            }
        with open(result_path, encoding="utf-8") as fh:
            return json.load(fh)
    except subprocess.TimeoutExpired:
        return {
            "status": "failure",
            "error_type": "TimeoutExpired",
            "message": f"MACE subprocess exceeded {_SUBPROCESS_TIMEOUT_S}s",
        }
    finally:
        for path in (job_path, result_path):
            try:
                os.unlink(path)
            except OSError:
                pass


def _mace_worker_inproc(job: dict) -> dict:
    """Execute a single MACE simulation directly in the caller's interpreter.

    This is the historical worker body. It is invoked by
    :mod:`chemgraph.mcp._mace_subrunner` inside the subprocess that
    :func:`_mace_worker` spawns, and is exposed here so callers that want
    direct in-process execution (e.g. local CLI runs without a Parsl
    backend) can still reach the same code path.
    """
    import json
    import tempfile

    from chemgraph.schemas.mace_parsl_schema import mace_input_schema
    from chemgraph.tools.parsl_tools import run_mace_core

    job = dict(job)

    remote_file = job.pop("remote_structure_file", None)
    if remote_file is not None:
        job["input_structure_file"] = remote_file
        if not os.path.isabs(job.get("output_result_file", "")):
            job["output_result_file"] = os.path.join(
                os.path.dirname(remote_file),
                job.get("output_result_file", "output.json"),
            )

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
                tmpdir,
                job.get("output_result_file", "output.json"),
            )

    output_file = job.get("output_result_file")
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    params = mace_input_schema(**job)
    result = run_mace_core(params)

    if inline is not None and isinstance(result, dict):
        out_file = job.get("output_result_file", "")
        if os.path.isfile(out_file):
            with open(out_file, encoding="utf-8") as fh:
                result["full_output"] = json.load(fh)

    return result


def _ls_remote_files(path: str) -> list[str]:
    """Backend-side helper: list non-directory entries in *path*."""
    return sorted(
        f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))
    )
