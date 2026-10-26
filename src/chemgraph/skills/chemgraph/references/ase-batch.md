# Write an ASE batch calculation

Use the existing `ASEInputSchema` and `run_ase_core` in a workspace Python script.
ChemGraph and the requested calculator must be installed in the compute environment.
Keep the agent and LLM calls on the submission host; run the calculation through PBS.
Read the `pbs-hpc` skill and its site reference for environment and launch settings.

If the structure is missing, first follow [local preparation](structure-preparation.md)
and use the returned artifact path.

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

Copy [calculate.py](../assets/calculate.py) into the run directory unchanged. Its
PBS host check precedes chemistry imports and rejects accidental execution on a
login node. It reads `input.json` from the working directory.

Fill the existing PBS template with site resources and environment setup, export
`CHEMGRAPH_LOG_DIR="$PWD"` after changing to the run directory, and launch
`exec /absolute/path/to/compute/python calculate.py`. Quote shell paths and use
JSON serialization for input values. For a single Polaris GPU calculation, launch
one process with `CUDA_VISIBLE_DEVICES=0` and `OMP_NUM_THREADS=8`. Source the compute
environment with nounset temporarily disabled if activation requires it, then
restore it; retain the site's proxy settings and set `TMPDIR=/tmp` before Python.
Validate script syntax without executing the calculation on the login node.

The script exits 0 for success, 2 for nonconvergence, and 1 for failure. Inspect
PBS stdout/stderr and the existing result JSON (`success`, `converged`,
`potential_energy`, and driver-specific data). Frequencies and trajectories use
the run directory. Early failures may leave only stderr; missing results or
scheduler history do not establish success. No separate summary schema is needed.
