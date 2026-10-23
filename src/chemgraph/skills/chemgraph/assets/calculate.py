"""Run input.json from the calculation directory inside a PBS allocation.

Exits 0 for success, 2 for nonconvergence, and 1 for failure.
"""

import json
import os
from pathlib import Path
import socket


def main():
    # The PBS host check precedes chemistry imports and rejects login nodes.
    nodefile = os.environ.get("PBS_NODEFILE")
    if not os.environ.get("PBS_JOBID") or not nodefile:
        raise RuntimeError("Submit this calculation through PBS.")
    nodes = {name.split(".")[0] for name in Path(nodefile).read_text().split()}
    if socket.gethostname().split(".")[0] not in nodes:
        raise RuntimeError("This host is not in the PBS allocation.")

    from chemgraph.schemas.ase_input import ASEInputSchema, ASEOutputSchema
    from chemgraph.tools.ase_core import _resolve_existing_path, run_ase_core

    params = ASEInputSchema.model_validate_json(Path("input.json").read_text())
    result = run_ase_core(params)
    print(json.dumps(result, indent=2))
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
