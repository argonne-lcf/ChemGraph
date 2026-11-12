# Single points and ground-state optimization

Use the supplied scientific settings. This reference covers ordinary molecular
single points and ground-state geometry optimization; inspect the installed
Gaussian documentation for other input forms or algorithms.

## Prepare an input

For an explicit-coordinate input, retain the Gaussian section order: Link 0
commands, route, blank line, title, blank line, charge/multiplicity, coordinates,
and a terminating blank line. Preserve any required basis/ECP, connectivity, or
constraint sections after the geometry. Check coordinate units instead of
silently treating every input as angstrom.

Use a new `%Chk` destination and explicit `%Mem` and `%NProcShared` values chosen
for the allocation. Check initialization settings or launch options for
conflicting resource overrides.

These are route fragments with placeholders, not executable inputs or method
recommendations:

```text
#p METHOD/BASIS SP
#p METHOD/BASIS Opt
```

Choose the requested operation and replace every placeholder. Retain the
project's additional settings, including solvent, symmetry, and convergence
options when specified. General basis/ECP inputs may need a different route and
additional sections.

For Cartesian inputs with freeze codes, preserve the atom label, code, and
remaining annotations when changing coordinates. A code of `-1` fixes an atom;
`0` permits optimization. Select constrained atoms from the user's/project's
instructions, not their element or position in an earlier study. Preserve
separate `ModRedundant` or other constraint definitions when present.
See the [molecular specification reference](https://www.conflex.co.jp/gaussian_support/molspec.php)
and [Opt reference](https://www.conflex.co.jp/gaussian_support/opt.php).

Before execution, check the geometry, atom order, charge/multiplicity, requested
method/basis and environment, constraints, output destinations, and consistency
between the Gaussian resource settings and allocation.

## Launch the application

For a shell-based installation, initialize Gaussian using the supplied setup
instructions, enter the chosen run directory, and set scratch. Replace the path
below with the verified executable; the variable is only a shell convenience:

```bash
GAUSSIAN_EXE=/absolute/path/to/g16
gaussian_status=0
"$GAUSSIAN_EXE" < input.gjf > output.log || gaussian_status=$?
# Perform any required stage-out here without losing gaussian_status.
exit "$gaussian_status"
```

This is an application fragment for the designated execution environment.
Adapt the executable and filenames to the installation and prepared input. Keep
stderr in the execution environment's captured error stream. Choose new output
and checkpoint paths before using redirection. If stage-out fails, report that failure as well and do
not mark the workflow complete. Handle submission and tracking through the
user's chosen execution mechanism.

## Inspect the result

Start with the echoed route, charge/multiplicity, geometry, and program banner.
They establish what actually ran. Inspect all intended sections of a Link1 or
concatenated output independently; an earlier success cannot validate a later
failed section.

For a single point, require normal termination of the relevant calculation,
successful electronic convergence for the requested method, and the requested
energy/property. An earlier `SCF Done` line is insufficient if a later step
failed. Report the energy definition and units; a single-point geometry is not
an optimized geometry.

For an ordinary optimization, inspect `Optimization completed`, the associated
force/displacement convergence table, and subsequent normal termination. The
usual table contains maximum/RMS force and maximum/RMS displacement checks.
Report the actual values and outcomes; if Gaussian uses a documented alternative
convergence condition, inspect that condition rather than silently applying a
different rule. A cycle limit, stopped optimization, or last coordinate table
alone does not establish convergence. See the
[Opt convergence documentation](https://www.conflex.co.jp/gaussian_support/opt.php).

Missing completion evidence means incomplete or unverified output. Determine
whether the job is still active through the execution mechanism before choosing
recovery. Retain explicit error messages and their context.

## Export coordinates and report limits

Identify the complete orientation table belonging to the converged geometry in
the relevant calculation section. Do not blindly export the final table in an
entire multi-step log. Check atom identities, count, order, finite coordinates,
and units; name the source log and table location in the export notes. Use a
distinct destination and retain the original input and log.

Report the constraints actually requested. When checking coordinate retention,
account for orientation changes and printed precision; do not claim exact
invariance solely from freeze flags. Optimization convergence does not by itself
establish an unconstrained local minimum. That characterization requires
appropriate frequency analysis and interpretation; do not add a frequency job
without it being part of the requested scientific task.
