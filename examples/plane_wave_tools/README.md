# Plane-wave DFT tools: run_qe and run_vasp

ChemGraph exposes two calculator-pinned tools for plane-wave DFT:

- **`run_qe`** - Quantum ESPRESSO (`pw.x`)
- **`run_vasp`** - VASP

Both are thin variants of `run_ase` with the calculator fixed to one engine, so
the LLM sees only that engine's parameters. They are registered automatically in
the `single_agent` workflow, and each is offered **only when its engine is
detected on the host** (binary + pseudopotentials), so an uninstalled engine is
never shown to the model.

[`run_plane_wave.py`](run_plane_wave.py) drives them end to end: an Argo-hosted
LLM reads a natural-language prompt, chooses the tool, fills in the input, and
the tool launches the real DFT engine through ASE.

> Scope note: this example proves the ChemGraph -> tool -> engine flow and the
> code path. The Quantum ESPRESSO path is runnable on Aurora (real `pw.x`); the
> VASP path is documented for a VASP-equipped host and has **not** been run on
> real VASP here (Aurora has no VASP binary). Treat the VASP section as the
> setup a VASP user would follow, not a validated run.

---

## Prerequisites

Assumes the DFT packages are already installed on your machine. You need three
things: (1) the DFT engine reachable by ASE, (2) a pseudopotential library, and
(3) an LLM endpoint (this example uses Argo via `argo-shim`).

### 1. Quantum ESPRESSO (`run_qe`)

ChemGraph builds `EspressoProfile(command=$ASE_ESPRESSO_COMMAND)` and ASE
appends `-in <input>` itself, so `ASE_ESPRESSO_COMMAND` must be the launch
**prefix only** (executable + launcher flags), with no `-in` / redirection.

**On Aurora** `pw.x` is prebuilt (no module file; add the bin dir to PATH):

```bash
# Vendor build + its runtime modules (see the build's own job.sub):
module load oneapi/release/2025.3.1 mpich/opt/develop-git.6037a7a hdf5/1.14.6

QE_BIN=/soft/applications/quantum_espresso/7.5-oneapi2025.0.5/cpu/bin/pw.x

# MPI launch prefix (102 ranks/node on Aurora CPU, 1 OpenMP thread each).
# ASE appends "-in <file>"; do NOT add -in / > here.
export ASE_ESPRESSO_COMMAND="mpiexec -np 102 -ppn 102 -d 1 $QE_BIN"
export OMP_NUM_THREADS=1

# Pseudopotentials: none are bundled. Point ESPRESSO_PSEUDO at your own set
# (e.g. SSSP). This gate must be set for run_qe to be offered.
export ESPRESSO_PSEUDO=$HOME/pseudo/sssp_efficiency
```

**On a generic cluster**: put `pw.x` on `PATH` (or set `ASE_ESPRESSO_COMMAND`)
and export `ESPRESSO_PSEUDO`.

```bash
export ASE_ESPRESSO_COMMAND="mpirun -np 4 pw.x"
export ESPRESSO_PSEUDO=/path/to/pseudopotentials
```

Download pseudopotentials matching your elements and XC functional. For the
bulk-Si default prompt you need a Si UPF in `$ESPRESSO_PSEUDO` (the SSSP PBE
efficiency set covers it).

### 2. VASP (`run_vasp`)

VASP is licensed. ASE accepts several launch mechanisms; set one, plus the
pseudopotential path:

```bash
# Launch command (any one of these ASE-recognized forms):
export ASE_VASP_COMMAND="mpirun -np 4 vasp_std"    # or:
# export VASP_COMMAND="srun vasp_std"
# ...or put vasp_std / vasp_gam on PATH

# POTCAR library root (contains potpaw_PBE/, potpaw_LDA/, ...).
export VASP_PP_PATH=/path/to/vasp/potcars
```

`run_vasp` is offered only when a VASP binary/command **and** `VASP_PP_PATH` are
both set. Aurora has no VASP binary, so this step is for a VASP-licensed host.

### 3. Argo LLM endpoint

`run_plane_wave.py` calls an LLM through Argo via `argo-shim`. Follow
[`../connecting_to_argo/README.md`](../connecting_to_argo/README.md) to bring up
the tunnel so ChemGraph can POST to `http://127.0.0.1:18085/argoapi/v1` from a
compute node. Then:

```bash
export ARGO_USER=<your.cels.login>          # required
# optional overrides:
# export ARGO_MODEL=argo:gpt-4.1-mini
# export ARGO_BASE=http://127.0.0.1:18085/argoapi/v1
```

You can point `ARGO_BASE` at any OpenAI-compatible endpoint; the Argo shim is
just what we validated on ALCF hardware.

---

## Running

From the repo root, inside your ChemGraph environment:

```bash
# Quantum ESPRESSO (runnable on Aurora):
python examples/plane_wave_tools/run_plane_wave.py --engine qe

# VASP (on a VASP-equipped host):
python examples/plane_wave_tools/run_plane_wave.py --engine vasp
```

Override the prompt to try your own chemistry:

```bash
QE_PROMPT="Compute the total energy of an isolated water molecule with Quantum ESPRESSO." \
  python examples/plane_wave_tools/run_plane_wave.py --engine qe
```

Expected: an `INFO` line showing the Argo model mapping, then the LLM calling
`run_qe` (or `run_vasp`), the DFT engine running through ASE, and ChemGraph
reporting the result. If the requested engine is not registered on the host, the
script exits early and prints exactly which environment variables are missing.

---

## How it fits together

```
prompt --> ChemGraph (single_agent) --> LLM picks run_qe / run_vasp
                                              |
                                     QEInputSchema / VaspInputSchema
                                     (calculator pinned to one engine)
                                              |
                                         run_ase_core --> ASE --> pw.x / VASP
```

`run_qe` / `run_vasp` are defined in
[`src/chemgraph/tools/ase_tools.py`](../../src/chemgraph/tools/ase_tools.py) and
their pinned input schemas in
[`src/chemgraph/schemas/plane_wave_input.py`](../../src/chemgraph/schemas/plane_wave_input.py).
Availability gating (which engines are offered) lives in
[`src/chemgraph/schemas/ase_input.py`](../../src/chemgraph/schemas/ase_input.py).
