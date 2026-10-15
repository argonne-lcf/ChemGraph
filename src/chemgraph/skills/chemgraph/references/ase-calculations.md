# Prepare and run an ASE calculation

## Prepare `input.json`

Use the supplied structure file, or follow [structure preparation](structure-preparation.md)
when coordinates need generation. Write an `ASEInputSchema` input with the
requested driver, calculator and scientific settings. This shows the input shape;
replace the placeholders and add the settings required by the selected calculation:

```json
{
  "input_structure_file": "<structure path>",
  "output_results_file": "result.json",
  "driver": "<requested driver>",
  "calculator": {"calculator_type": "<requested calculator>"}
}
```

Include the requested model, device, charge, multiplicity and optimization
parameters where applicable. For MACE-Polar, see [its input example](mace-polar.md).
`fmax` is in eV/Å. `vib` already optimizes before frequencies; `ir` also requires
calculator dipoles; `thermo` uses the supplied temperature and pressure.

When field names, types or defaults are needed, inspect the installed schema
without running a calculation:

```python
import json
from chemgraph.schemas.ase_input import ASEInputSchema

print(json.dumps(ASEInputSchema.model_json_schema(), indent=2))
```

An already-loaded `run_ase` tool also exposes this schema through its `params`
argument. Calculator options depend on the environment's installed dependencies.
For a remotely submitted job, inspect the schema in the intended remote compute
environment. If that environment is unavailable, ask the user to provide the
correct environment and access instructions, or the schema exported from it.
Do not infer remote calculator availability from the local schema or substitute
a calculator because its dependencies are absent locally.

## Run the calculation

Copy the shared [calculate.py](../assets/calculate.py) into the run directory
alongside `input.json`. It validates the input with `ASEInputSchema`, calls
`run_ase_core`, and checks the saved result. ChemGraph and the requested calculator
dependencies must be installed in the execution environment. The runner sets
`CHEMGRAPH_LOG_DIR` to its current directory before importing chemistry code;
relative inputs and outputs belong to that run directory.

Launch from the run directory with the selected Python executable:

```sh
/absolute/path/to/python calculate.py
```

Run locally when requested. For separate job execution, supply the execution
workflow with this command, the working directory, `input.json`, the runner,
structures, model weights, and any supporting files. Include the required
software environment, device/thread requirements, expected outputs, and the
exit-code meanings below. Use run-relative or compute-visible paths in inputs;
the execution workflow establishes file visibility and enforces host restrictions.
For preparation-only checks, inspect syntax or the input schema without executing
the calculation.

## Interpret results

The runner exits 0 for success/convergence, 2 for nonconvergence, and 1 for failure.
`run_ase_core` returns `status` and `message`; the result JSON records `success`,
`converged`, `error`, `potential_energy` (eV), and driver-specific results.
Distinguish calculation success from optimization convergence. Use returned
artifact paths, including `trajectory_file` when present. Early failures may
leave only stderr; scheduler completion alone does not establish scientific success.
