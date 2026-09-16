"""Run one ASE calculation inside an existing PBS allocation."""

import argparse
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import socket
import sys


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def _require_allocation():
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile:
        raise RuntimeError("Submit this calculation through PBS.")
    nodes = {
        name.split(".")[0].lower()
        for name in Path(nodefile).read_text(encoding="utf-8").split()
    }
    if not nodes:
        raise RuntimeError("PBS_NODEFILE contains no allocated hosts.")
    if socket.gethostname().split(".")[0].lower() not in nodes:
        raise RuntimeError("This host is not in the PBS allocation.")


def main(argv=None):
    """Return 0 for success, 1 for failure, or 2 for unconverged optimization."""
    parser = _ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="ASEInputSchema JSON file")
    result = None
    try:
        args = parser.parse_args(argv)
        _require_allocation()
        # Imports and calculations may print diagnostics; reserve stdout for JSON.
        with redirect_stdout(sys.stderr):
            from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema
            from chemgraph.tools.ase_core import run_ase_core

            params = ASEInputSchema.model_validate_json(
                Path(args.input).read_text(encoding="utf-8")
            )
            result = run_ase_core(params)
            code = 1
            if result.get("status") == "success":
                output = ASEOutputSchema.model_validate_json(
                    Path(result["results_file"]).read_text(encoding="utf-8")
                )
                if not output.success:
                    raise RuntimeError(output.error or "ASE output reports failure.")
                requires_optimization = params.driver in {"opt", "vib", "thermo", "ir"}
                code = 2 if requires_optimization and not output.converged else 0
    except Exception as exc:
        result = {
            **(result or {}),
            "status": "failure",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        code = 1
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
