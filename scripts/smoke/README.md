# ChemGraph execution-layer smoke tests

Self-contained scripts that exercise each ExecutionBackend live and emit
`[PASS]` / `[FAIL]` per check. Exit code is `0` only if every check passes
(`2` if required env vars are missing → "skip"). Use them for one-shot
validation after install, after a rebase, or before running real workloads.

These are *not* pytest tests — they hit live infrastructure (process pools,
PBS allocations, Globus Compute endpoints, Globus Transfer). The mocked
unit suite still lives at `tests/test_execution.py`.

## Script matrix

| Script | Runs where | Backends | Live deps |
|--------|------------|----------|-----------|
| [`smoke_local.py`](smoke_local.py) | laptop | `local` | none |
| [`smoke_globus_compute.py`](smoke_globus_compute.py) | laptop | `globus_compute` | live GC endpoint |
| [`smoke_globus_transfer.py`](smoke_globus_transfer.py) | laptop | `GlobusTransferManager` (+ optional `globus_compute` MCP) | Globus collections on both ends |
| [`smoke_parsl_in_job.py`](smoke_parsl_in_job.py) | inside `qsub -I` on Polaris/Aurora | `parsl` | PBS allocation |
| [`smoke_ensemble_launcher_in_job.py`](smoke_ensemble_launcher_in_job.py) | inside `qsub -I` on Polaris/Aurora | `ensemble_launcher` (managed + client-only) | PBS allocation, `ensemble_launcher` built from source |

`_smoke_utils.py` holds shared helpers (`SmokeReporter`, picklable trivial
callables). `water.xyz` is the shared 3-atom fixture.

## Environment-variable matrix

| Variable | Required by | Notes |
|----------|-------------|-------|
| `GLOBUS_COMPUTE_ENDPOINT_ID` | `smoke_globus_compute.py`, `smoke_globus_transfer.py --with-mcp` | UUID printed by `globus-compute-endpoint start chemgraph-<system>` |
| `GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID` | `smoke_globus_transfer.py` | Globus Connect Personal collection on the laptop |
| `GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID` | `smoke_globus_transfer.py` | HPC collection UUID (ALCF data portal) |
| `GLOBUS_TRANSFER_DESTINATION_BASE_PATH` | `smoke_globus_transfer.py` | e.g. `/eagle/projects/MyProj/staging` (Polaris), `/flare/projects/MyProj/staging` (Aurora) |
| `COMPUTE_SYSTEM` | `smoke_parsl_in_job.py`, `smoke_ensemble_launcher_in_job.py` | `polaris` or `aurora` |
| `PBS_NODEFILE` | both in-job scripts | Set automatically by PBS inside `qsub` — the scripts abort if missing |
| `CG_SMOKE_DEVICE` | optional, MACE device override | Defaults: `cuda` (Polaris/Globus), `xpu` (Aurora) |

## Running

### Laptop only (no creds)

```bash
source .cg_env/bin/activate
python scripts/smoke/smoke_local.py             # ~5s + first-run MACE model download
python scripts/smoke/smoke_local.py --quick     # ~3s, skips MACE
```

### Laptop → live Globus Compute endpoint

```bash
export GLOBUS_COMPUTE_ENDPOINT_ID="<uuid-from-endpoint-start>"
export COMPUTE_SYSTEM=polaris   # or aurora
python scripts/smoke/smoke_globus_compute.py
python scripts/smoke/smoke_globus_compute.py --amqp 443   # Aurora (5671 blocked)
```

### Laptop → live Globus Transfer

```bash
export GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID="<laptop-collection-uuid>"
export GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID="<hpc-collection-uuid>"
export GLOBUS_TRANSFER_DESTINATION_BASE_PATH=/eagle/projects/MyProj/staging
python scripts/smoke/smoke_globus_transfer.py            # transfer only
python scripts/smoke/smoke_globus_transfer.py --with-mcp # also dispatch MACE ensemble in remote-path mode
```

First run triggers an OAuth flow; the token caches at
`~/.globus/chemgraph_transfer_tokens.json` for subsequent runs.

### Inside a PBS allocation on Polaris

```bash
qsub -I -A <proj> -l select=1 -l walltime=01:00:00 -q debug -l filesystems=home:eagle
# (now on the compute node)
module load conda
conda activate base
source ~/chemgraph/venv/bin/activate
export COMPUTE_SYSTEM=polaris
cd ~/chemgraph/ChemGraph

python scripts/smoke/smoke_parsl_in_job.py
python scripts/smoke/smoke_ensemble_launcher_in_job.py --mode managed
```

### Inside a PBS allocation on Aurora

```bash
qsub -I -A <proj> -l select=1,walltime=01:00:00 -q debug -l filesystems=home:flare
module load frameworks
source ~/chemgraph/venv/bin/activate
export COMPUTE_SYSTEM=aurora
cd ~/chemgraph/ChemGraph

python scripts/smoke/smoke_parsl_in_job.py --device xpu
python scripts/smoke/smoke_ensemble_launcher_in_job.py --mode managed --device xpu
```

### EnsembleLauncher client-only mode

Exercises `EnsembleLauncherBackend(client_only=True, ...)` introduced in
commit `bc54083c`. Requires two shells on the same compute node:

```bash
# Shell A — start the orchestrator
cd $PBS_O_WORKDIR
python -m ensemble_launcher \
    --system $COMPUTE_SYSTEM \
    --checkpoint-dir $PBS_O_WORKDIR/el_ckpt \
    --node-id 0

# Shell B — connect this client to it
python scripts/smoke/smoke_ensemble_launcher_in_job.py \
    --mode client-only \
    --checkpoint-dir $PBS_O_WORKDIR/el_ckpt \
    --node-id 0
```

The client-only run leaves the orchestrator in Shell A running; stop it
there with `Ctrl-C` when done.

## See also

- `scripts/hpc_setup/README.md` — install ChemGraph + Globus Compute endpoint on Polaris/Aurora
- `scripts/hpc_setup/e2e_test_runbook.md` — tier-by-tier manual runbook (these smoke scripts are the automation around Tiers 1, 2, and the gap tests)
- `scripts/globus_compute_example/` — tutorial-style demonstrations (longer-form than the smoke scripts)
- `src/chemgraph/execution/` — the production code paths these scripts call
