"""Run an ASE calculation in its own process and artifact directory."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def require_allocation():
    """Reject login-node execution, including inherited allocation variables."""
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile:
        raise RuntimeError("Run this calculation through PBS, not on a login node.")
    nodes = {name.split(".")[0] for name in Path(nodefile).read_text().split()}
    if socket.gethostname().split(".")[0] not in nodes:
        raise RuntimeError("This host is not listed in PBS_NODEFILE.")


def run_calculation(input_file, *, require_pbs=False):
    """CLI calculation body. Call in a dedicated process, not a worker thread."""
    if require_pbs:
        require_allocation()
    from ase.io import write

    from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema
    from chemgraph.tools.ase_core import (
        _resolve_existing_path,
        _resolve_path,
        atomsdata_to_atoms,
        run_ase_core,
    )

    payload = json.loads(Path(input_file).read_text())
    if not payload.get("calculator") or not payload.get("output_results_file"):
        raise ValueError("Specify calculator and output_results_file explicitly.")
    params = ASEInputSchema(**payload)
    if params.driver not in {"energy", "dipole", "opt", "vib", "ir", "thermo"}:
        raise ValueError(
            "Specify an ASE driver: energy, dipole, opt, vib, ir, or thermo."
        )
    params.input_structure_file = str(
        Path(_resolve_existing_path(params.input_structure_file)).resolve()
    )
    results = Path(_resolve_path(params.output_results_file)).resolve()
    if results.suffix != ".json":
        raise ValueError("output_results_file must end in .json.")
    params.output_results_file = str(results)
    summary_file = results.parent / "run_summary.json"
    if results == summary_file or results.exists() or summary_file.exists():
        raise FileExistsError(
            "Use a fresh calculation directory; existing results are preserved."
        )
    results.parent.mkdir(parents=True, exist_ok=True)
    # This process owns all derived frequency, mode, spectrum and log paths.
    os.environ["CHEMGRAPH_LOG_DIR"] = str(results.parent)
    summary = {
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "hostname": socket.gethostname(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "driver": params.driver,
        "calculator": params.calculator.model_dump(mode="json"),
        "results_file": str(results),
    }
    exit_code = 1
    try:
        calculator = params.calculator
        if calculator.calculator_type.startswith("mace_"):
            if not calculator.model:
                raise ValueError(
                    "Stage a local MACE model and specify its path explicitly."
                )
            model = Path(calculator.model).expanduser().resolve(strict=True)
            with model.open("rb") as stream:
                summary["model_sha256"] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
            calculator.model = str(model)
            summary["calculator"] = calculator.model_dump(mode="json")
        if getattr(calculator, "device", None) == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but is unavailable; no CPU fallback."
                )
        result = run_ase_core(params)
        summary.update(result)
        if results.exists():
            output = ASEOutputSchema.model_validate_json(results.read_text())
            if output.final_structure is not None:
                final_file = results.parent / "final.xyz"
                write(
                    final_file, atomsdata_to_atoms(output.final_structure), format="xyz"
                )
                summary["final_structure_file"] = str(final_file)
            if result.get("status") == "success" and output.success:
                exit_code = 0 if output.converged else 2
                if not output.converged:
                    summary["status"] = "not_converged"
        elif result.get("status") == "success":
            raise RuntimeError("Calculation returned success without a result file.")
    except Exception as exc:
        summary.update(
            status="failure", error_type=type(exc).__name__, message=str(exc)
        )
    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    summary["exit_code"] = exit_code
    summary["artifacts"] = sorted(
        str(p) for p in results.parent.iterdir() if p.is_file()
    )
    summary_file.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return exit_code


def run_pbs_calculation(job):
    """Run an MCP task in a child process, isolating ASE's process-wide paths.

    The caller supplies a fresh output directory per calculation. A marker
    prevents a repeated/concurrent submission from overwriting its artifacts.
    """
    from chemgraph.tools.ase_core import _resolve_existing_path, _resolve_path

    try:
        job = dict(job)
        job["input_structure_file"] = str(
            Path(_resolve_existing_path(job["input_structure_file"])).resolve()
        )
        result_path = Path(_resolve_path(job["output_results_file"])).resolve()
        job["output_results_file"] = str(result_path)
        root = result_path.parent
        root.mkdir(parents=True, exist_ok=True)
        request = root / "ase_input.json"
        summary_file = root / "run_summary.json"
        log_file = root / "calculation.log"
        if result_path in (request, summary_file) or any(
            path.exists() for path in (result_path, request, summary_file, log_file)
        ):
            raise FileExistsError("Use a fresh output directory for every calculation.")
        with (root / "run.started").open("x"):
            pass
        request.write_text(json.dumps(job, indent=2, default=str) + "\n")
        with log_file.open("w") as log:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "chemgraph.tools.ase_runner",
                    "--input",
                    str(request),
                    "--require-pbs",
                ],
                cwd=root,
                env={**os.environ, "CHEMGRAPH_LOG_DIR": str(root)},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if summary_file.exists():
            summary = json.loads(summary_file.read_text())
            if completed.returncode != summary.get("exit_code"):
                summary.update(
                    status="failure",
                    message=f"Process exited {completed.returncode}; inspect the calculation log.",
                )
            return {
                **summary,
                "summary_file": str(summary_file),
                "log_file": str(log_file),
            }
        return {
            "status": "failure",
            "message": f"Calculation exited {completed.returncode} without a summary.",
            "results_file": str(result_path),
            "log_file": str(log_file),
        }
    except Exception as exc:
        return {
            "status": "failure",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON using ASEInputSchema")
    parser.add_argument(
        "--require-pbs",
        action="store_true",
        help="Require a PBS compute node before calculation setup",
    )
    args = parser.parse_args()
    try:
        return run_calculation(args.input, require_pbs=args.require_pbs)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
