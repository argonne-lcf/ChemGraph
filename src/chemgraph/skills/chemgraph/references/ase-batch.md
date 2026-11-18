# Write an ASE batch calculation

Use the existing `ASEInputSchema` and `run_ase_core` in a workspace Python script.
ChemGraph and the requested calculator must be installed in the compute environment.
Keep the agent and LLM calls on the host driving the workflow; run the calculation
inside PBS, through [pbs-hpc](../../pbs-hpc/SKILL.md) or [iri-hpc](../../iri-hpc/SKILL.md).

If the structure is missing, first follow [local preparation](structure-preparation.md)
and use the returned artifact path.

Write `input.json` with the user's structure, calculator/model, and scientific
settings in a fresh run directory. For direct PBS, use absolute paths visible on
compute nodes and put the result JSON in that shared directory. For IRI, use
run-relative calculation paths such as `water.xyz` and `result.json` for files
staged with the run. JobSpec paths remain absolute remote paths. Never pass
laptop paths to the remote calculation.

Use the requested calculator and a supported model name, or omit `model` to use
that calculator's default. A model file path is optional for custom or pre-staged
checkpoints. This Polaris example uses MACE-Polar. For Crux set `device` to `cpu`. For
an XPU-capable MACE installation on Aurora use `xpu` and `"default_dtype": "float32"`.

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
    "model": "polar-1-m",
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
temperature and pressure. Preserve these choices. Stage explicitly supplied
checkpoint files or use files already accessible on compute nodes; for IRI,
checkpoints staged with the run can use run-relative paths. Named pretrained
models need no user-supplied file: the calculator downloads missing weights during
initialization inside the allocation. This needs compute-node network access or
an existing cache; the ALCF templates set the
[proxy](../../pbs-hpc/references/polaris.md#internet-proxy-login-and-compute-nodes).
Do not execute the calculation on a login node to download or initialize a model.

Copy [calculate.py](../assets/calculate.py) into the run directory unchanged. Its
PBS host check precedes chemistry imports and rejects accidental execution on a
login node. It reads `input.json` from the working directory.

Use the [site PBS template](../../pbs-hpc/SKILL.md#prepare-and-track-a-job)
or IRI's `iri_prepare.py` for the environment, proxy, and log directory.
Write `input.json` with JSON serialization, not hand-built strings.

Before staging/submission, parse `input.json`, inspect scientific settings and
path mappings, check the structure, and run `bash -n` on the launch script.
Local `ASEInputSchema` validation depends on locally installed calculator extras:
an unavailable-calculator error on a laptop does not establish that the remote
environment lacks it. Do not install the remote calculator stack on the laptop
solely for validation or substitute a calculator to make validation pass. Report
which checks ran and defer environment-dependent schema validation to the
configured compute environment. The unchanged `calculate.py` validates the schema
and initializes the calculator only after its compute-node guard passes inside
the allocation. Syntax/input checks do not establish runtime success.

The script exits 0 for success, 2 for nonconvergence, and 1 for failure. Inspect
PBS stdout/stderr and the existing result JSON (`success`, `converged`,
`potential_energy`, and driver-specific data). Frequencies and trajectories use
the run directory. Early failures may leave only stderr; missing results or
scheduler history do not establish success.
