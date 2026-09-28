"""Run an existing ASE input.json inside a PBS compute allocation."""

import json
import os
from pathlib import Path
import socket
import sys


def main():
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile:
        raise RuntimeError("Submit this calculation through PBS.")
    nodes = {node.split(".")[0] for node in Path(nodefile).read_text().split()}
    if socket.gethostname().split(".")[0] not in nodes:
        raise RuntimeError("This host is not in the PBS allocation.")
    os.environ["CHEMGRAPH_LOG_DIR"] = str(Path.cwd())
    print(
        json.dumps(
            {
                "compute_hostname": socket.gethostname(),
                "python": sys.executable,
                "pbs_job_id": os.environ["PBS_JOBID"],
                "cwd": str(Path.cwd()),
            }
        )
    )
    from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema
    from chemgraph.tools.ase_core import _resolve_existing_path, run_ase_core

    params = ASEInputSchema.model_validate_json(Path("input.json").read_text())
    result = run_ase_core(params)
    print(json.dumps(result))
    if result.get("status") != "success":
        return 1
    output = ASEOutputSchema.model_validate_json(
        Path(_resolve_existing_path(params.output_results_file)).read_text()
    )
    if not output.success:
        return 1
    return 0 if output.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
