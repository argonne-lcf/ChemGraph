---
name: gaussian
description: Prepare Gaussian single-point and ground-state optimization inputs, launch calculations in the selected environment, and inspect convergence and geometries.
license: Apache-2.0
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Gaussian calculations

Use this skill for Gaussian input preparation, execution, and output inspection.
Read [single points and optimization](references/optimization.md) for the
currently supported procedures. TDDFT, recovery, and export procedures are not
included in this version.

## Establish the calculation

Take the scientific setup from the user's request or the project's explicit
instructions: geometry and units, method, basis/ECP, charge and multiplicity,
environment/solvent, constraints, and requested task. Inspect supplied inputs
before changing them. Ask only for missing essential information or conflicting
settings; do not silently choose a method, solvent, spin state, or constraints.
Examples illustrate syntax and do not establish project defaults.

Preserve atom order and atom annotations when preparing related calculations.
Check that the charge/multiplicity and basis coverage fit the supplied model.
Record the input geometry source and any deliberate changes to the setup.

## Prepare the execution environment

Reuse the Gaussian executable path and initialization instructions supplied by
the user or project. If none are supplied, check whether Gaussian is already
configured (for example, `command -v g16` in a shell). If unresolved, ask for the
executable or installation path and any environment setup needed to use it.
Do not guess installation directories or initialization commands. Verify the
executable after setup and reuse that configuration for subsequent tasks.

Resolve the working directory, available CPU/memory resources, and scratch
location from the task or environment; ask for missing essential choices.
Match Gaussian shared-memory threads and memory to those resources, leaving
room for process overhead. Execution/submission arrangements come from the
user's environment and are separate from this software guidance.

Choose a new run directory and checkpoint/output names. Verify scratch capacity
and visibility; set `GAUSS_SCRDIR` after software initialization, which may change
the environment. Preserve needed checkpoints and results on durable storage
before temporary scratch is reclaimed. Use the execution resources designated
for the calculation.

## Execute and inspect

Preparation alone does not authorize a run. A request to run the calculation
does; use the existing authorization without repeatedly asking. Inspect the
prepared input and launch command before execution. Preserve the Gaussian exit
status through logging, cleanup, and stage-out.

Inspect the relevant calculation section for normal/error termination and the
requested scientific result. Process exit and a printed energy alone do not
establish success. Incomplete output may belong to an active calculation.

Report the input/output paths, geometry source, scientific settings, actual
software revision from the output banner, execution identity when available,
convergence evidence, and unresolved limitations. Keep energies and coordinates
labeled with units.
The optimization reference describes manual checks; this version includes no
inspection helper.
