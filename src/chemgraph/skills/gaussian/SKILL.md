---
name: gaussian
description: Prepare and run Gaussian single points and ground-state optimizations; diagnose input, checkpoint, SCF, geometry-convergence, and scratch failures.
license: Apache-2.0
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Gaussian calculations

Use the user's/project's geometry, method, basis/ECP, charge, multiplicity,
solvent, and constraints. Ask for missing essentials; examples are syntax aids,
not scientific defaults. Inspect an existing input before rebuilding it: XYZ
conversion can discard frozen-atom flags and other Gaussian annotations.

## Select the procedure

- New single point or optimization: read [input and result checks](references/optimization.md).
  Start from the [optimization template](assets/optimization.gjf.template) for
  explicit Cartesian coordinates and a built-in basis.
- Reuse a checkpoint, resume an optimization, or diagnose failure: read
  [checkpoints and troubleshooting](references/troubleshooting.md). `Guess=Read`,
  `Geom=AllCheck`, and `Opt=Restart` perform different operations.
- TDDFT, excited-state optimization, and property-export workflows need separate
  procedures; do not apply ground-state acceptance checks to them.

## Resolve Gaussian setup once

Reuse the supplied executable path and initialization instructions. Otherwise
check `command -v g16`; if unresolved, ask for the Gaussian executable/install
path and required initialization. Do not invent an installation path. Verify
the executable after setup and reuse this configuration.

Resolve the run directory, CPU/memory allocation, and scratch location. Set
`GAUSS_SCRDIR` after initialization and inspect any `%RWF` overrides. Use unique
checkpoint/output names for independent runs. Match `%NProcShared` and `%Mem`
to the allocation, allowing memory overhead; shell defaults do not override
explicit Link 0 settings. Execution arrangements come from the environment.

Before running, resolve all template tokens and inspect the input sections.
Preserve the process exit status through logging and stage-out. After running,
check each intended job step: `SCF Done`, optimization completion, and normal
termination answer different questions. A missing terminator may mean the job
is still running; check its execution status before attempting recovery.

Report the Gaussian revision, input/log/checkpoint paths, settings actually
used, and convergence evidence with units. Preserve the original log and
checkpoint when preparing a changed attempt.
