"""Backend worker functions for MACE MCP tools.

This module intentionally contains no FastMCP/CGFastMCP objects or tool
decorators. Parsl/dill serializes worker functions by walking their module
globals, so backend workers must live outside modules that contain FastMCP's
runtime-generated argument classes.
"""

import os


def _mace_worker(job: dict) -> dict:
    """Execute a single MACE simulation on a backend worker."""
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
