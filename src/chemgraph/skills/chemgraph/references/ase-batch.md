# Write an ASE batch calculation

Use the existing `ASEInputSchema` and `run_ase_core` in a workspace Python script.
ChemGraph and the requested calculator must be installed in the compute environment.
Keep the agent and LLM calls on the submission host; run the calculation through PBS.
Read the `pbs-hpc` skill and its site reference for environment and launch settings.

Write `input.json` with the user's structure, calculator/model, and scientific
settings. Use absolute host paths visible on compute nodes, and a fresh shared
directory for each calculation. Put the result JSON in that directory. For example,
this MACE-Polar input assumes a staged local model and its optional dependencies;
it is not a default calculator choice for other requests:

```json
{
  "input_structure_file": "/shared/run/water.xyz",
  "output_results_file": "/shared/run/result.json",
  "driver": "vib",
  "optimizer": "bfgs",
  "fmax": 0.01,
  "steps": 200,
  "calculator": {
    "calculator_type": "mace_polar",
    "model": "/shared/models/polar-1-m.model",
    "device": "cuda",
    "default_dtype": "float64",
    "charge": 0,
    "multiplicity": 1
  }
}
```

`opt` optimizes geometry; `vib` already optimizes before calculating frequencies.
Use one `vib` job for optimization plus frequencies. `ir` additionally requires
calculator dipoles; `thermo` computes ideal-gas thermochemistry using the supplied
temperature and pressure. Preserve these choices and stage model files before
submission. Do not execute the calculation on a login node to download or
initialize a model.

Write `calculate.py` using this example. The PBS host check precedes chemistry
imports and rejects accidental execution on a login node:

```python
import json
import os
from pathlib import Path
import socket

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
    raise SystemExit(1)
output = ASEOutputSchema.model_validate_json(
    Path(_resolve_existing_path(params.output_results_file)).read_text()
)
if not output.success:
    raise SystemExit(1)
raise SystemExit(0 if output.converged else 2)
```

Fill the existing PBS template with site resources and environment setup, export
`CHEMGRAPH_LOG_DIR="$PWD"` after changing to the run directory, and launch
`exec /absolute/path/to/compute/python calculate.py`. Quote shell paths and use
JSON serialization for input values. For a single Polaris GPU calculation, launch
one process with `CUDA_VISIBLE_DEVICES=0` and `OMP_NUM_THREADS=8`. Source the compute
environment with nounset temporarily disabled if activation requires it, then
restore it; retain the site's proxy settings and set `TMPDIR=/tmp` before Python.
Validate script syntax without executing the calculation on the login node.

The example exits 0 for success, 2 for nonconvergence, and 1 for failure. Inspect
PBS stdout/stderr and the existing result JSON (`success`, `converged`,
`potential_energy`, and driver-specific data). Frequencies and trajectories use
the run directory. Early failures may leave only stderr; missing results or
scheduler history do not establish success. No separate summary schema is needed.
