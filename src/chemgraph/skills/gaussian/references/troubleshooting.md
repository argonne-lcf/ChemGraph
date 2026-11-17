# Checkpoints and troubleshooting

## Checkpoint reuse

First inspect the source log and checkpoint provenance: molecule, atom order,
charge/spin, model chemistry, and completed stage. File existence alone does
not prove it contains the records the next calculation needs. Never copy a
checkpoint while the original calculation is still writing it.

| Intent | Gaussian option | Input consequence |
| --- | --- | --- |
| Use saved coordinates for a new calculation | `Geom=Check` | Supply a title and charge/multiplicity; omit coordinates. Specify the new calculation's route. |
| Use saved coordinates, charge/multiplicity, and title | `Geom=AllCheck` | Omit all three sections; after the route's blank line, supply only any additional input required by other keywords. Do not insert a habitual `0 1`. |
| Reuse orbitals as the initial guess | `Guess=Read` | Reads the wavefunction guess; it does not by itself read the geometry or resume the optimizer. Check compatibility with the requested state/model. |
| Resume an interrupted optimization with its saved state | `Opt=Restart` | Requires usable optimization records. The route uses `Opt` with the relevant original optimization options and `Restart`; model chemistry and geometry come from the checkpoint, with no title/molecule input. |

See [Geom](https://www.conflex.co.jp/gaussian_support/geom.php),
[Guess](https://www.conflex.co.jp/gaussian_support/guess.php), and
[Opt/Restart](https://www.conflex.co.jp/gaussian_support/opt.php).
If changing the method or constraints, prepare a new optimization from a chosen
geometry instead of treating it as a transparent restart.

Preserve a source checkpoint with distinct destinations:

```text
%OldChk=source.chk
%Chk=next.chk
```

`%OldChk` copies source records into the current checkpoint at the start of the
step; subsequent writes go to `%Chk`. Resolve both paths from the actual run
directory. Independent jobs must not share a writable checkpoint. This copies
data; route keywords still decide which records to use. See
[Link 0 commands](https://www.conflex.co.jp/gaussian_support/link0.php).

## Resource settings that are easy to misread

- `%Mem` without a suffix counts **8-byte words**. Use explicit byte units
  such as `GB`; `MW` and `MB` are different. Leave memory for process overhead.
- Explicit Link 0 settings override corresponding defaults. Changing
  `GAUSS_MDEF` does not replace `%Mem` already in the input. Inspect command-line
  settings, environment, and a working-directory `Default.Route` when effective
  settings differ from expectations.
- `%NProcShared` selects shared-memory processors; increasing it does not by
  itself distribute a job across machines.
- Set `GAUSS_SCRDIR` after the installation's initialization. If `%RWF` is
  present, inspect its actual file locations too. Check free space, quota, and
  permissions there; increasing `%Mem` is not a general disk-error fix.

Sources: [Link 0](https://www.conflex.co.jp/gaussian_support/link0.php) and
[defaults and precedence](https://www.conflex.co.jp/gaussian_support/defroute.php).

## Diagnose before changing the route

Read the lines preceding termination and the stage that was running. A link
executable name such as `l9999.exe` identifies where execution stopped, not a
unique cause. The checks below are diagnostic starting points, not automatic
retry recipes. Preserve the failed input/log and record each deliberate change.

| Symptom or stage | First check / targeted response |
| --- | --- |
| Route parse error (`QPErr`) | Check spelling, parentheses, unresolved template tokens, and plain ASCII punctuation. Electronic convergence options cannot fix a route parser error. |
| Unexpected EOF or molecule-reading error | Check route/title/molecule blank lines and the selected `Geom` mode. Check charge/multiplicity, coordinate fields, and required trailing sections. |
| Basis/ECP read error | Check element coverage and basis/ECP blocks, not just the route. `GenECP` requires both kinds of data. |
| SCF fails to converge at one geometry | Inspect geometry, charge, spin, and the SCF history. `SCF=(MaxCycle=N)` changes electronic iterations. For supported HF/DFT cases, `SCF=XQC` adds a quadratic-convergence fallback; QC is unavailable for restricted open-shell calculations. Do not blindly append it to every failure. |
| Geometry step limit reached after converged SCFs | `Opt=(MaxCyc=N)` controls geometry steps; increasing the SCF limit will not help. Inspect force/displacement trends before extending the optimization. |
| Coordinate construction failure or oscillating optimization | Inspect distorted/linear angles, atom ordering, and constraints. Preserve constraint semantics when considering a coordinate-system change; do not silently remove freezes or loosen convergence thresholds. |
| Write failure, quota error, or external kill | Inspect scratch/checkpoint paths and the execution system's exit/resource evidence. Preserve surviving files before cleanup; missing Gaussian termination alone cannot distinguish a running job from an externally killed one. |

Keyword behavior: [SCF](https://www.conflex.co.jp/gaussian_support/scf.php),
[Opt](https://www.conflex.co.jp/gaussian_support/opt.php),
[input sections](https://www.conflex.co.jp/gaussian_support/input.php), and
[general basis/ECP](https://www.conflex.co.jp/gaussian_support/gen.php).
