# Write an ASE batch calculation

Use `ASEInputSchema` and `run_ase_core` directly; preparing files does not require
loading a calculation tool or reading its source. Read `pbs-hpc` and its selected
site reference. ChemGraph and the requested calculator must exist in the compute
environment. Preserve the user's settings; EMT below is only a CPU smoke example.
For MACE-Polar, read [its input example](mace-polar-batch.md).

For IRI submission without a shared compute filesystem, follow `hpc-batch` and
its runner/staging workflow instead of the direct PBS script below. Transfer
run-relative structures, models and results; keep scientific settings unchanged.

## Inputs

Use a fresh run directory visible to compute nodes. Given explicit coordinates,
write `hydrogen.xyz` directly; if coordinates need generation, follow
[local preparation](structure-preparation.md) first.

```xyz
2
H2, H-H distance 1.0 Angstrom
H 0 0 0
H 0 0 1
```

Write `input.json`. Relative paths resolve under `CHEMGRAPH_LOG_DIR`, set to the
run directory below; absolute paths must be compute-visible host paths.

```json
{
  "input_structure_file": "hydrogen.xyz",
  "output_results_file": "result.json",
  "driver": "opt",
  "optimizer": "bfgs",
  "fmax": 0.05,
  "steps": 100,
  "calculator": {"calculator_type": "emt"}
}
```

`fmax` is in eV/Å. `vib` already optimizes before frequencies; `ir` also requires
calculator dipoles; `thermo` uses the supplied temperature and pressure.

## Compute script

Write `calculate.py`. Keep the allocation check before chemistry imports:

```python
import json
import os
from pathlib import Path
import socket
import sys

nodefile = os.environ.get("PBS_NODEFILE")
if not os.environ.get("PBS_JOBID") or not nodefile:
    raise RuntimeError("Submit this calculation through PBS.")
nodes = {name.split(".")[0] for name in Path(nodefile).read_text().split()}
if socket.gethostname().split(".")[0] not in nodes:
    raise RuntimeError("This host is not in the PBS allocation.")
print(json.dumps({"compute_hostname": socket.gethostname(),
                  "python": sys.executable, "pbs_job_id": os.environ["PBS_JOBID"]}))

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

## PBS script

Write `job.pbs` for this one-node Polaris CPU example. Replace project, environment,
Python and absolute output-log placeholders with the user's values; explicit
placeholders may remain for a prepare-only request, but resolve them before submission.
Keep directives before commands. Use shell quoting for paths and JSON serialization
for input values. The script starts in the directory used to submit the job.

```bash
#!/bin/bash -l
#PBS -N chemgraph-h2-emt-smoke
#PBS -A YOUR_PROJECT
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=00:05:00
#PBS -l filesystems=home:eagle
#PBS -o /absolute/shared/run/job.stdout
#PBS -e /absolute/shared/run/job.stderr

set -euo pipefail
cd "${PBS_O_WORKDIR:?PBS submission directory is required}"
set +u
source /absolute/path/to/environment.sh
set -u
export CHEMGRAPH_LOG_DIR="$PWD"
export TMPDIR=/tmp
export OMP_NUM_THREADS=1
exec /absolute/path/to/environment/bin/python calculate.py
```

## Validate without running

From the run directory, use an available ChemGraph Python environment for this
validation only; it does not execute `calculate.py`, create a calculator or submit:

```sh
bash -n job.pbs && python - <<'PYVALIDATE'
import ast
from pathlib import Path
from ase.io import read
from chemgraph.schemas.ase_input import ASEInputSchema
from chemgraph.tools.ase_core import _resolve_existing_path

ast.parse(Path("calculate.py").read_text())
params = ASEInputSchema.model_validate_json(Path("input.json").read_text())
atoms = read(_resolve_existing_path(params.input_structure_file))
assert atoms.get_chemical_symbols() == ["H", "H"]
assert atoms.positions.tolist() == [[0, 0, 0], [0, 0, 1]]
print("Validated shell, Python, ASE input and H2 coordinates; calculation not run.")
PYVALIDATE
```

Set `CHEMGRAPH_LOG_DIR` to the run directory for validation too. Adapt the structure
checks to the requested molecule. If the local environment lacks the requested
calculator schema, report incomplete validation; never substitute EMT to pass it.

## Results contract

Exit codes: 0 for success/convergence, 2 for nonconvergence, 1 for failure.
The return dictionary has `status` and `message`; failures may include `error_type`.
The result JSON has `success`, `converged`, `error`, `potential_energy` (eV), and
driver-specific data. Check both return status and saved results.
Optimization of a multi-atom file creates `<input-stem>_opt.traj` beside the result
JSON: this example produces `hydrogen_opt.traj`; no trajectory parameter is needed.
The return dictionary reports `trajectory_file` when exported. Frequency artifacts
use the run directory. Early failures may leave only stderr; PBS completion alone
is not scientific success. For prepare-only work, report validation and file paths;
result JSON and trajectories will exist only after calculation.
