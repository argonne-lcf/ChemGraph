# MACE-Polar input

Use this input for a requested MACE-Polar calculation with the matching optional
dependencies installed in the execution environment. When no model is specified,
ChemGraph selects `polar-1-m`; MACE reuses cached weights or downloads them
automatically. A user-provided weights path is optional. Preserve an explicitly
requested model name or path.

Adapt [the ASE calculation](ase-calculations.md); keep the user's model, device, dtype,
charge, multiplicity, driver and optimization settings. This example optimizes
water and computes frequencies in one `vib` job:

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

For remote execution, the cache or supplied path must be visible in the compute
environment. If weights are uncached and that environment cannot download them,
download and stage the selected checkpoint ahead of execution, then use its
compute-visible path. Ask for help only when the required access or model cannot
be resolved. Include the selected device and software requirements in the
application handoff. Preparing inputs does not require initializing the model
or running the calculation.
