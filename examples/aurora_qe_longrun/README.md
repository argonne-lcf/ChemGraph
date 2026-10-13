# Aurora QE long-run cap to resume validation

This example validates ChemGraph's **long-running-calculation** support on **real
Quantum ESPRESSO DFT** on ALCF Aurora: a calculation that would exceed the job's
wall-clock allocation self-terminates at an ASE optimizer-step boundary, writes a
durable *partial geometry* plus manifest, and a later `chemgraph resume` continues
from that partial and skips recomputing from scratch.

The core principle is **enforcement by deadline**. The run makes no attempt to
guess whether the next step will fit in the remaining time. It runs steps until a
wall-clock deadline (`CHEMGRAPH_ALLOCATION_DEADLINE` / `_SECONDS`, minus a safety
`_MARGIN`) is crossed at a step boundary, then stops cleanly with a resumable
artifact. The cap is *soft*: it is checked only between optimizer steps and leaves
an in-flight SCF untouched, so the margin must exceed one SCF step's wall time.

> The fast, portable, CI-run version of these checks lives in the repo's unit
> tests (`tests/test_cap*.py`, `tests/test_manifest.py`,
> `tests/test_allocation_cap.py`, `tests/test_resume_injection.py`,
> `tests/test_mace_cap.py`) using the in-process EMT/MACE calculators. This
> directory is the **real-DFT, HPC counterpart**. It needs Aurora plus QE plus an
> LLM endpoint and therefore cannot run in CI, yet it proves the same seam
> end-to-end on a subprocess DFT engine, which is the ultimate target.

## What each route exercises

| File | Route | Stack exercised |
|------|-------|-----------------|
| `qe_cap_driver.py` | **Route 1, calc layer** | `run_ase_core` calls QE directly, with no LLM/agent. Proves the cap fires at a step boundary, writes a resumable partial, and a resume continues to the same energy. |
| `qe_agent_e2e.py` | **Route 2, full agent** | real LLM to LangGraph `single_agent` to `run_ase` ToolNode (JSON-serialized return) to `run_ase_core` to QE, plus the manifest hook and `chemgraph resume`. Proves the layers Route 1 bypasses: tool-arg unwrapping, JSON tool-message parsing, `tool_call_id` correlation, clear-pending-on-success (M2), and log_dir adoption on resume (M3). |

Route 2 is the real production path. Route 1 is the isolation check you run first
to confirm QE plus pseudopotentials plus the cap seam work before adding LLM
variance.

### Periodic and non-periodic: two systems, two code paths

Route 1 covers **both** DFT k-point regimes, because they run different code:

- **Bulk Si (`smoke`/`cap`/`resume`)** is fully periodic (`pbc=[T,T,T]`), so the
  configured Monkhorst-Pack mesh is used verbatim. This is the original seam check.
- **H2O molecule (`mol_smoke`/`mol_cap`/`mol_resume`)** is non-periodic
  (`pbc=[F,F,F]`, no cell), so it exercises the molecule path added on this branch:
  `is_nonperiodic(atoms)` then `atoms.center(vacuum=…)` (a finite box for the
  plane-wave basis) then **`K_POINTS gamma`** (a Monkhorst-Pack mesh is
  meaningless for an isolated molecule). Bulk Si leaves this path untouched, so the
  H2O stages are what validate the molecule fix on live `pw.x`. `mol_smoke` and
  `mol_cap` grep the generated `espresso.pwi` for `K_POINTS gamma` plus a finite
  cell, and `mol_resume` asserts the relaxed geometry lands in the cross-code
  plane-wave-PBE water band (see *Validated results*).

## Files

- `qe_cap_driver.py`: Route 1 driver. Bulk-Si stages `smoke` | `cap` | `resume`;
  H2O-molecule stages `mol_smoke` | `mol_cap` | `mol_resume`.
- `qe_agent_e2e.py`: Route 2 driver, stages `run1` | `resume`.
- `run_qe_cap.pbs`: PBS batch script for Route 1 bulk-Si (runs all three bulk-Si
  stages `smoke` | `cap` | `resume` in one allocation).
- `run_qe_mol.pbs`: PBS batch script for Route 1 H2O molecule (runs all three
  `mol_*` stages in one allocation, covering the non-periodic gamma/centering path).
- `run_agent_e2e.pbs`: PBS batch script for Route 2 bulk Si (run1 **then** resume
  in one allocation, as two separate processes so the resume genuinely re-adopts
  state).
- `run_agent_mol.pbs`: PBS batch script for Route 2 H2O molecule (same run1
  **then** resume flow with `QE_SYSTEM=h2o`, driving the non-periodic
  gamma/centering path through the full agent stack).

## Prerequisites (Aurora)

1. **ChemGraph installed** in a Python environment; point `PY` in the PBS script
   at it.
2. **Quantum ESPRESSO `pw.x`**. Aurora ships a prebuilt binary at
   `/soft/applications/quantum_espresso/…/bin/pw.x` (see `QE_BIN` in the scripts).
3. **Pseudopotentials**: a directory pointed to by `ESPRESSO_PSEUDO` (default
   `$HOME/qe_pseudo`) holding, for the bulk-Si stages, `Si.UPF` (PBE), and for the
   H2O molecule stages the matching pslibrary USPP PBE H/O pseudos
   `H.pbe-rrkjus_psl.1.0.0.UPF` and `O.pbe-n-rrkjus_psl.1.0.0.UPF`. Fetch the H/O
   pair (same family, functional, and type as the Si pseudo: pslibrary 1.0.0
   ultrasoft PBE) from the QE pseudopotential library:
   ```bash
   cd "$ESPRESSO_PSEUDO"
   base=https://pseudopotentials.quantum-espresso.org/upf_files
   curl -sSLO $base/H.pbe-rrkjus_psl.1.0.0.UPF
   curl -sSLO $base/O.pbe-n-rrkjus_psl.1.0.0.UPF
   ```
4. **An LLM endpoint** (Route 2 only). These scripts use the ALCF inference API
   (`openai/gpt-oss-120b`) reached through the ALCF HTTP proxy. Put a valid ALCF
   access token at `~/.alcf_token` (the driver reads it into `ALCF_ACCESS_TOKEN`;
   see the repo's ALCF auth notes). Aurora compute nodes have no direct outbound
   network, hence the `*_PROXY` block in `run_agent_e2e.pbs`.

## Run it

```bash
cd examples/aurora_qe_longrun

# Route 1, bulk Si (no LLM): prove the cap seam on real QE in isolation.
qsub run_qe_cap.pbs

# Route 1, H2O molecule (no LLM): prove the non-periodic gamma/centering path.
qsub run_qe_mol.pbs

# Route 2, bulk Si (full agent): run1 (capped) then resume, in one allocation.
qsub run_agent_e2e.pbs

# Route 2, H2O molecule (full agent): the non-periodic path through the agent stack.
qsub run_agent_mol.pbs
```

The Route 2 scripts write each job to a fresh per-job work dir under the submit
directory (`agent_e2e_<jobid>` / `agent_mol_<jobid>`), so a resubmission stays
clean of a prior attempt. The Route 1 scripts reuse a fixed run dir (`cap_run` /
`mol_run`) across resubmissions; the cap stage overwrites the partial-path
pointer each time. Each script echoes `RUN1 OK` / `RESUME OK` (Route 2) or
`CAP OK` / `RESUME OK` (Route 1) plus the manifest JSON. The drivers `assert` every invariant, so a
non-zero exit signals a real regression.

## Validated results

> ### Scope: these runs validate the mechanism
>
> The runs below validate the **flow**: the cap-to-resume seam, and for H2O the
> non-periodic Gamma/centering code path on live `pw.x`. They make no claim about
> the physical accuracy of the numbers. Cutoffs, vacuum padding, and convergence
> thresholds are sized for a fast validation run that fits one debug allocation,
> well short of a converged production calculation. The geometry band is a coarse
> sanity guard that catches a broken run (a blown-up geometry from a wrong k-mesh
> or a zero cell); read it as a coarse pass/fail guard; a converged accuracy figure is a separate goal. A
> physically-rigorous convergence study is a planned future update of this example.

### Route 1 and Route 2, bulk Si

Measured on the Aurora `debug` queue (account `ChemGraph`), 2026-07-28. System:
rattled 2-atom Si diamond, `ecutwfc=25 Ry`, `2×2×2` k-mesh, PBE, ~1 s/SCF at 8 MPI
ranks.

- **Route 1 (calc layer):** cap fired at an optimizer step boundary, left
  `*_opt.partial.xyz` plus `*_opt.restart.json` plus `*_opt.traj`, and exited
  cleanly with no walltime kill. Resume from the partial converged to the **same**
  energy as a single uncapped opt, **-308.187965 eV**, so the cap preserves the
  result.
- **Route 2 (full agent):** run1 capped mid-opt (24.2 s), manifest
  `status=capped` with a PENDING step whose `input_structure_file` is the partial;
  resume (fresh process, new default log_dir) adopted run1's log_dir
  from the session DB (**M3**), cleared pending and reset status to
  `running` (**M2**), continued from the partial, and converged to
  **-308.18796684 eV**.
- **Route 2 under langgraph 1.x:** the same flow re-run after the
  upstream dependency major-bump (langgraph 1.2.9 / langchain 1.x) behaves
  identically and converges to **-308.18796440 eV** (about 1e-6 eV of BFGS noise
  away from the other runs).

### Route 1, H2O molecule (non-periodic Gamma path), 2026-07-30

System: ASE `molecule("H2O")` (`pbc=[F,F,F]`, cell-less) centered into a 6 Å
vacuum box by `ase_core`; `ecutwfc=50`, `ecutrho=400 Ry`, PBE, pslibrary USPP H/O
pseudos, 8 MPI ranks, about 14 s/SCF. All three `mol_*` stages passed (`rc=0`):

- **`mol_smoke`:** single-point **-473.2192 eV**. The generated `espresso.pwi`
  carries `K_POINTS gamma` plus a finite `CELL_PARAMETERS`, so the centering
  reached the writer and the new `is_nonperiodic` to `center(vacuum)` to gamma path
  drove real pw.x.
- **`mol_cap`:** BFGS capped at a step boundary after 2 steps (49.9 s of an
  about 50 s effective window), leaving `h2o_cap_opt.partial.xyz` plus
  `.restart.json` plus `.traj`; the `.pwi` again shows `K_POINTS gamma` plus a
  finite cell.
- **`mol_resume`:** uncapped resume from the partial converged in 7 BFGS steps
  (fmax 2.71 down to 0.006) to **-473.2196 eV**. Relaxed geometry
  **r(O-H) = 0.9709 / 0.9710 Å, ∠(H-O-H) = 104.40°**, inside the cross-code
  plane-wave-PBE band (0.96-1.00 Å, 102-106°; compare JDFTx PBE about 0.98 Å /
  103.7°, experiment 0.9572 Å / 104.52°).

The cap left a resumable partial and the resume continued to a converged,
physically-sensible water geometry, so the molecule cap-to-resume flow works
end-to-end on real DFT.

### Route 2, H2O molecule (full agent stack), 2026-07-30

The same non-periodic Gamma/centering path as the Route 1 H2O run, driven this
time through the whole agent stack (real LLM to `single_agent` to `run_ase` to
`run_ase_core` to QE, plus the manifest hook and `chemgraph resume`), so it
covers the molecule fix and the layers Route 1 bypasses at once. Same system and
settings as the Route 1 H2O run (`molecule("H2O")` centered into a 6 Å vacuum
box, `ecutwfc=50`, `ecutrho=400 Ry`, PBE, pslibrary USPP H/O pseudos, 8 MPI
ranks). Both stages passed (`rc=0`):

- **run1 (capped):** the agent called `run_ase` once, BFGS capped at a step
  boundary after 4 steps (66.6 s effective wall-clock), single-point
  **-473.2142 eV** for the partial. The manifest recorded `status=capped` with a
  PENDING step whose `input_structure_file` is the partial geometry, and the
  generated `espresso.pwi` carries `K_POINTS gamma` plus a `CELL_PARAMETERS`
  block (the agent path asserts the block is present; the finite-volume cell
  check is done by the Route 1 driver).
- **resume (fresh process):** adopted run1's `log_dir` from the session DB
  (**M3**), cleared the pending step and reset `status` (**M2**), continued from
  the partial, and converged to **-473.2196 eV** (matching the Route 1 H2O
  resume). The resume `espresso.pwi` again shows `K_POINTS gamma` plus a
  `CELL_PARAMETERS` block, and the relaxed geometry is **r(O-H) = 0.9709 / 0.9710 Å,
  ∠(H-O-H) = 104.43°**, inside the cross-code plane-wave-PBE band (0.96-1.00 Å,
  102-106°).

So the full agent stack reproduces the calc-layer physics on the molecule path:
the same converged energy and a water geometry that matches the Route 1 result to
within BFGS noise, with the cap and resume driven entirely through the LLM tool
layer.

### Resume-prompt hardening (a real lesson, baked into `qe_agent_e2e.py`)

An early Route 2 resume attempt failed because the prompt asked the LLM to echo
the **full absolute** partial-geometry path. `gpt-oss-120b` then (a) dropped the
required single `params` wrapper and (b) mistyped the path, burning every
recursion with zero successful tool calls, so the clear-on-success (M2) path never
even ran. The state machine behaved correctly (nothing succeeded, so nothing was
cleared); the failure was in the *prompt*.

The fix lives on the prompt side:

- Hand the model only the **bare basename** (`si_agent_opt.partial.xyz`), which
  leaves no long path to mistype. It still resolves because resume adopts the prior
  session's `log_dir` into `CHEMGRAPH_LOG_DIR` (M3) and `run_ase`'s input path is
  resolved against that dir. (This rescues a bare name; a mistyped absolute path is
  still on the model to get right.)
- Spell out the single-`params`-argument tool contract explicitly.

A healthy Route 2 log still shows a benign `Error invoking tool 'run_ase' … params:
Field required` line. That is the model's **own** self-correcting retry: the first
call omitted the `params` wrapper and the next message re-sent it correctly. It is
expected, and it is separate from any framework error.

## Porting to another HPC

The cap-to-manifest-to-resume machinery itself is fully portable (that is what the
in-process unit tests prove on any laptop). Only this HPC *harness* is
Aurora-specific. To run it elsewhere, edit the marked sections of the PBS scripts:

- **Scheduler directives:** the `#PBS -A/-q/-l …` lines. On a Slurm site, rewrite
  them as `#SBATCH …` and submit with `sbatch`.
- **Environment modules:** the `module load oneapi/… mpich/… hdf5/…` lines become
  whatever provides an MPI plus QE toolchain on your machine.
- **`QE_BIN`:** path to your `pw.x`.

> **Note on version pins.** The exact QE build path in `QE_BIN`
> (`.../quantum_espresso/7.5-oneapi2025.0.5/...`) and the `mpich/opt/develop-git.<hash>`
> snapshot module are point-in-time Aurora values and will drift as the facility
> updates its software stack. They are pinned deliberately so a run is reproducible,
> and `set -euo pipefail` makes a stale path fail loudly at launch. Refresh both
> against the current `module avail` / `/soft/applications` on the day you run.
- **`ESPRESSO_PSEUDO`:** your pseudopotential directory (needs `Si.UPF`, plus the
  H/O pseudos for the molecule stages).
- **`PY`:** a Python with ChemGraph installed.
- **LLM plus proxy** (Route 2): the `*_PROXY` block is only needed on networks that
  firewall compute nodes (like Aurora). If your LLM endpoint is directly reachable,
  drop it. To use a different model or endpoint, change `QE_LLM_MODEL` and the auth
  the driver loads (for example `OPENAI_API_KEY` plus an `api.openai.com` model in
  place of the ALCF token).

## Future direction: unattended auto-resubmit

Within one allocation (one `qsub` window), resume is already automatic: when the
cap fires, the agent continues from the saved partial on its own, and the example
PBS scripts also show resume as a fresh process in the same allocation. The step
that is still manual is starting the *next* allocation: once a `qsub` window is
exhausted the compute node is released, and ChemGraph does not submit scheduler
jobs, so continuing across a fresh allocation means running `qsub` again by hand.
A later layer would submit that next allocation automatically, so a calculation
longer than any single allocation runs to completion across many allocations with
no human in the loop. The design for that layer, built on the DOE IRI Facility
API, is written up in [`IRI_INTEGRATION.md`](IRI_INTEGRATION.md), including the
endpoint and field mapping, why it is additive on top of the current manifest (no
schema break), and what live-facility access it would need. It is a design note;
none of it is implemented here.
