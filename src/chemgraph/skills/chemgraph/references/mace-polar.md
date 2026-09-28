# MACE-Polar input

Use this input only for a requested MACE-Polar calculation with staged local
weights and the matching optional dependencies installed in the execution environment.
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
    "model": "/shared/models/polar-1-m.model",
    "device": "cuda",
    "default_dtype": "float64",
    "charge": 0,
    "multiplicity": 1
  }
}
```

Make model weights available before execution. Include the selected device and
software requirements in the application handoff. Preparing inputs does not
require initializing the model or running the calculation.
