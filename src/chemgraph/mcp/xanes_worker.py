"""Backend worker functions for XANES MCP tools.

This module intentionally contains no FastMCP/CGFastMCP objects or tool
decorators, keeping worker functions safe for Parsl/dill serialization.
"""

import subprocess
from pathlib import Path

from chemgraph.schemas.xanes_schema import xanes_input_schema


def run_xanes_single(params: xanes_input_schema) -> dict:
    """Run a single FDMNES calculation on a backend worker."""
    from chemgraph.tools.xanes_tools import run_xanes_core

    result = run_xanes_core(params)
    if isinstance(result, dict):
        result.setdefault("status", "success")
        return result
    return {"status": "success", "result": result}


def _xanes_ensemble_worker(item: dict) -> dict:
    """Execute one prepared FDMNES run on the backend."""
    from chemgraph.tools.xanes_tools import extract_conv

    run_dir = item["run_dir"]
    fdmnes_exe = item["fdmnes_exe"]
    meta = {
        "structure": item.get("structure"),
        "run_dir": run_dir,
        "z_absorber": item.get("z_absorber"),
    }

    stdout_path = Path(run_dir) / "fdmnes_stdout.txt"
    stderr_path = Path(run_dir) / "fdmnes_stderr.txt"
    try:
        with open(stdout_path, "w", encoding="utf-8") as out, open(
            stderr_path,
            "w",
            encoding="utf-8",
        ) as err:
            proc = subprocess.run(
                [fdmnes_exe],
                cwd=run_dir,
                stdout=out,
                stderr=err,
                check=False,
            )
        if proc.returncode != 0:
            return {
                **meta,
                "status": "failure",
                "error_type": "FDMNESExitCode",
                "message": f"FDMNES exited with code {proc.returncode}",
                "returncode": proc.returncode,
            }
    except Exception as e:
        return {
            **meta,
            "status": "failure",
            "error_type": type(e).__name__,
            "message": f"FDMNES launch failed: {e}",
        }

    try:
        conv_data = extract_conv(run_dir)
        return {
            **meta,
            "status": "success",
            "n_conv_files": len(conv_data),
        }
    except Exception as e:
        return {
            **meta,
            "status": "failure",
            "error_type": type(e).__name__,
            "message": f"Post-processing failed: {e}",
        }
