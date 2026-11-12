# Single points and ground-state optimization

## Use the input template

Copy [optimization.gjf.template](../assets/optimization.gjf.template) to a new
`input.gjf` in the run directory. Replace every `{{TOKEN}}` using the requested
calculation; Gaussian does not substitute these tokens. `EXTRA_ROUTE` may be
empty or contain the supplied solvent/other route options. `MEMORY` needs a
unit, such as `GB`; `NPROC` is an integer. Choose a new checkpoint filename.

`COORDINATES` is the molecular coordinate block, without an XYZ atom-count or
comment header. This template assumes Cartesian coordinates in angstrom;
convert other units explicitly or supply the appropriate `Units` option. Use
a single plain-text title line. Retain the blank lines after the route, title,
and coordinates, including the final blank line. For a single point, replace
`Opt` with `SP`. The resulting file should contain Link 0 settings, one route,
title, charge/multiplicity, and the complete geometry, with no `{{...}}` left.
See Gaussian's [input structure](https://www.conflex.co.jp/gaussian_support/input.php)
and [molecular specification](https://www.conflex.co.jp/gaussian_support/molspec.php).

This is a starting form, not a universal input generator:

| Input feature | Required adaptation |
| --- | --- |
| `Geom=Check` or `Geom=AllCheck` | Change the sections as described in [checkpoint reuse](troubleshooting.md#checkpoint-reuse); do not leave the template geometry in place. |
| `Gen` / `GenECP` | Append the supplied basis blocks; `GenECP` also reads an ECP section. A route keyword alone is insufficient. Preserve block separators and element coverage. |
| Frozen atoms | Retain the optional freeze field: `C -1 x y z` fixes that atom; `C 0 x y z` allows it to move. Do not strip these fields when replacing coordinates. |
| `Opt=ModRedundant` | Preserve its separate constraint block after the molecular specification. Atom references are numbered from 1, excluding dummy atoms; reordering atoms can constrain the wrong bond. |
| Connectivity or other read options | Retain the additional sections in the order required by the combined keywords. A misplaced blank line changes what Gaussian reads. |

Basis/ECP syntax: [Gen and GenECP](https://www.conflex.co.jp/gaussian_support/gen.php).
Freeze fields and numbering: [molecular specification](https://www.conflex.co.jp/gaussian_support/molspec.php)
and [Geom/ModRedundant](https://www.conflex.co.jp/gaussian_support/geom.php).

For an ordinary all-electron molecule, check charge/spin consistency:
`N = sum(atomic numbers) - charge`; `N - (multiplicity - 1)` must be nonnegative
and even. This detects impossible inputs, not the scientifically correct spin
state. ECP, ghost-atom, fragment, and multilayer inputs need their own checks.

## Launch in the configured environment

After the supplied initialization, enter the run directory and set scratch.
Use a verified executable and unused output name. This shell fragment preserves
Gaussian's failure status even if results must subsequently be copied:

```bash
GAUSSIAN_EXE=/absolute/path/to/g16
gaussian_status=0
"$GAUSSIAN_EXE" < input.gjf > output.log || gaussian_status=$?
# Stage out required results here; report a stage-out failure separately.
exit "$gaussian_status"
```

Keep stderr in the execution environment's captured error stream. Consult
[resource and failure checks](troubleshooting.md) if execution fails; repeating
the same input does not resolve a syntax, checkpoint, or storage problem.

## Read the result by job step

Inspect the echoed route, charge/multiplicity, and revision banner first.
For `--Link1--`, concatenated logs, or combined `Opt Freq` jobs, associate evidence
with the intended calculation. A successful earlier step cannot validate a
later failed step; report completed and failed stages separately.

| Evidence | What it establishes |
| --- | --- |
| `SCF Done` | An SCF energy at that geometry, not completion of an optimization or later correlated-energy calculation. |
| `Optimization completed` and its force/displacement table | Geometry convergence under the requested criteria. Read the actual maximum/RMS force and displacement values; Gaussian has alternative convergence conditions, so four `YES` entries are not a universal rule. |
| `Normal termination` | The relevant Gaussian job step finished; also require its requested scientific result. |
| Last geometry or energy in a truncated log | A partial result, unless completion can be established from the relevant step. Check whether execution is still active. |

For a single point, require the requested method's convergence, final energy or
property, and normal termination. For an optimization, require convergence
evidence and termination of that stage. `Opt=Tight` makes the geometry criteria
stricter; it is not a remedy for exceeding a step limit. See the
[optimization documentation](https://www.conflex.co.jp/gaussian_support/opt.php).

Export the complete geometry associated with the converged step, recording its
log location, atom order, and units. Gaussian's standard orientation can rotate
or translate coordinates: direct component-wise comparison with the input can
falsely suggest frozen atoms moved. Preserve original atom annotations and
constraints when creating a new input; an orientation table does not replace
them. Optimization convergence alone does not establish a local minimum;
frequency analysis and interpretation would be a separate scientific check.
