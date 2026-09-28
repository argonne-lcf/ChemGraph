# MACE-Polar batch input

Use this input only for a requested MACE-Polar calculation with staged local
weights and the matching optional dependencies installed on compute nodes.
Adapt [the ASE batch runner](ase-batch.md); keep the user's model, device, dtype,
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
    "model": "/shared/models/polar-1-m.model",
    "device": "cuda",
    "default_dtype": "float64",
    "charge": 0,
    "multiplicity": 1
  }
}
```

Stage model weights before submission. Do not initialize or download models by
running the calculation on a login node. For one Polaris GPU process, use the
single-GPU settings in the `pbs-hpc` Polaris reference.
