# ChemGraph execution-layer demonstration scripts

Real-chemistry demos that exercise each `ExecutionBackend` end-to-end.
A 5-molecule library (H2O, CH4, NH3, CO2, ethanol) is screened for
thermochemistry with MACE-MP (`driver="thermo"` → optimize geometry +
vibrational frequencies + ideal-gas thermo at 298.15 K). Each script
writes a CSV of electronic energy, enthalpy, entropy, Gibbs free
energy per molecule and prints a fixed-width summary table.

## Workloads (`--workload`)

The harness is registry-driven (`_demo_chemistry.py`); pick the tool with
`--workload`:

| Workload | Tool | Items | Extra needed | Notes |
|----------|------|-------|--------------|-------|
| `thermo` (default) | MACE-MP | molecule fixtures | (base) | original 5-molecule screen |
| `ase` | general ASE | molecule fixtures | (base) | selectable `--calculator` |
| `fairchem` | FairChem/UMA | molecule fixtures | `.[uma]` | `--model-name` (uma-s-1p1); GPU preferred |
| `graspa` | gRASPA GCMC | CIF paths (`--graspa-cifs`) | (HPC binary) | shared-FS / HPC only |
| `pacmof2` | PACMOF2 charges | CIF paths (`--pacmof2-cifs`) | `.[pacmof2]` + source install | `--net-charge`; CPU-only; shared-FS / HPC only |

`fairchem` reuses the molecular thermo table; `pacmof2` prints a per-element
charge summary. `graspa` and `pacmof2` take CIF paths reachable on the compute
node, not the molecule fixtures.

These complement `scripts/smoke/`:

| Directory | Purpose | Pass criterion |
|-----------|---------|---------------|
| `scripts/smoke/` | Regression validators on a trivial water payload | Exit 0 with every `[PASS]` |
| `scripts/demo/` | Realistic chemistry showcases | Useful property table; demos *fail loud* but their value is the output, not a green check |

## Layout

```
scripts/demo/
├── README.md                                       (this file)
├── _demo_chemistry.py                              shared helpers (workload, formatting, agent prompt)
├── structures/                                     5 .xyz fixtures (~50 lines each)
│   ├── water.xyz  methane.xyz  ammonia.xyz  co2.xyz  ethanol.xyz
├── demo_local_direct.py                            laptop, no LLM, no HPC
├── demo_local_agent.py                             laptop, LLM, no HPC
├── demo_globus_compute_direct.py                   laptop, no LLM, live GC endpoint
├── demo_globus_compute_agent.py                    laptop, LLM, live GC endpoint
├── demo_globus_transfer_direct.py                  laptop, no LLM, Globus Transfer + GC
├── demo_globus_transfer_agent.py                   laptop, LLM, Globus Transfer + GC
├── demo_parsl_in_job_direct.py                     inside qsub -I on Polaris/Aurora, no LLM
├── demo_parsl_in_job_agent.py                      inside qsub -I, LLM
├── demo_ensemble_launcher_in_job_direct.py         inside qsub -I, no LLM
└── demo_ensemble_launcher_in_job_agent.py          inside qsub -I, LLM
```

Direct demos call `chemgraph.execution.config.get_backend()` and
`backend.submit_batch(...)` directly. Agent demos spawn
`python -m chemgraph.mcp.mace_mcp_hpc` as a stdio subprocess and drive
it with a ChemGraph LLM agent over `langchain-mcp-adapters`.

## Environment-variable matrix

| Variable | Required by | Notes |
|----------|-------------|-------|
| `GLOBUS_COMPUTE_ENDPOINT_ID` | `demo_globus_compute_*`, `demo_globus_transfer_*` | UUID from `globus-compute-endpoint start chemgraph-<system>` |
| `GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID` | `demo_globus_transfer_*` | Globus Connect Personal on the laptop |
| `GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID` | `demo_globus_transfer_*` | HPC collection UUID (ALCF data portal) |
| `GLOBUS_TRANSFER_DESTINATION_BASE_PATH` | `demo_globus_transfer_*` | e.g. `/eagle/projects/MyProj/staging` |
| `COMPUTE_SYSTEM` | `demo_parsl_in_job_*`, `demo_ensemble_launcher_in_job_*` | `polaris` or `aurora` |
| `PBS_NODEFILE` | both in-job demos | Set automatically inside `qsub` — demos abort if missing |
| `CG_AMQP_PORT=443` | optional, Aurora | Use when outbound 5671 is blocked |
| LLM API key (e.g. `OPENAI_API_KEY`) | all `*_agent.py` | Match the `--model` flag |

## Running

### Laptop, no creds

```bash
source .cg_env/bin/activate
python scripts/demo/demo_local_direct.py
# ~20s for the 5 molecules on CPU; writes demo_local_out/{demo_local.csv,*_thermo.json}
```

Sample output:
```
=== Local backend thermo screen (cpu) ===
molecule       energy/eV    enthalpy/eV      S/(eV/K)          G/eV   #freqs    wall/s   conv
---------------------------------------------------------------------------------------------
water           -13.7861       -13.1063      0.001958      -13.6900        9       3.0   True
methane         -23.1669       -21.8802      0.001931      -22.4559       15       3.6   True
ammonia         -18.9970       -17.9888      0.001996      -18.5839       12       3.3   True
co2             -22.5459       -22.1320      0.002209      -22.7906        9       2.9   True
ethanol         -46.2767       -44.0648      0.002820      -44.9056       27       3.3   True
```

### Laptop + LLM

```bash
export OPENAI_API_KEY=...
python scripts/demo/demo_local_agent.py --model gpt-4o-mini
```

Agent will call `run_mace_single` 5 times via the MCP subprocess and
respond with a markdown table.

### Laptop → live Globus Compute endpoint

```bash
export GLOBUS_COMPUTE_ENDPOINT_ID="<uuid>"
export COMPUTE_SYSTEM=polaris   # for logging
python scripts/demo/demo_globus_compute_direct.py            # ~5-15 min first run (model download on remote)
python scripts/demo/demo_globus_compute_agent.py --model gpt-4o-mini
```

For Aurora add `--device xpu --amqp-port 443`.

### Laptop → Globus Transfer + Globus Compute

```bash
export GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID="<laptop-collection-uuid>"
export GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID="<hpc-collection-uuid>"
export GLOBUS_TRANSFER_DESTINATION_BASE_PATH=/eagle/projects/MyProj/staging
python scripts/demo/demo_globus_transfer_direct.py
python scripts/demo/demo_globus_transfer_agent.py --model gpt-4o-mini
```

The direct demo stages the 5 `.xyz` fixtures, then runs MACE in
*remote-path* mode (worker reads from the staged dir, no inline
embedding). The agent demo asks the LLM to call `transfer_files` and
then `run_mace_ensemble` itself.

Remote-path mode has one quirk: `_mace_worker` only attaches
`full_output` back to the caller when an `inline_structure` is set
(see `src/chemgraph/mcp/mace_mcp_hpc.py:127-131`). So in
`demo_globus_transfer_direct.py` the printed table will have blank
thermo columns — the full JSON results sit on the HPC under
`<remote_directory>/<molecule>_thermo.json`. Pull them back with a
follow-up Globus Transfer if needed.

### Inside a PBS allocation on Polaris

```bash
qsub -I -A <proj> -l select=1 -l walltime=01:00:00 -q debug -l filesystems=home:eagle
# Now on the compute node:
module load conda
conda activate base
source ~/chemgraph/venv/bin/activate
export COMPUTE_SYSTEM=polaris
cd ~/chemgraph/ChemGraph
python scripts/demo/demo_parsl_in_job_direct.py
python scripts/demo/demo_ensemble_launcher_in_job_direct.py
```

### Inside a PBS allocation on Aurora

```bash
qsub -I -A <proj> -l select=1,walltime=01:00:00 -q debug -l filesystems=home:flare
module load frameworks
source ~/chemgraph/venv/bin/activate
export COMPUTE_SYSTEM=aurora
cd ~/chemgraph/ChemGraph
python scripts/demo/demo_parsl_in_job_direct.py --device xpu
python scripts/demo/demo_ensemble_launcher_in_job_direct.py --device xpu
```

### Inside a PBS allocation on Crux (CPU-only)

```bash
qsub -I -A <proj> -l select=1 -l walltime=01:00:00 -q debug -l filesystems=home:eagle
cd /lus/eagle/projects/ChemGraph/thang/ChemGraph

bash scripts/demo/run_crux_demo.sh                       # Parsl + EL, all 5 molecules
bash scripts/demo/run_crux_demo.sh --molecules water methane
bash scripts/demo/run_crux_demo.sh --parsl-only
bash scripts/demo/run_crux_demo.sh --el-only
```

The wrapper activates `.cg_crux_hpc/`, exports `COMPUTE_SYSTEM=crux`, and runs
`demo_parsl_in_job_direct.py` then `demo_ensemble_launcher_in_job_direct.py`
with `--device cpu`. CSVs land in `$PBS_O_WORKDIR/demo_{parsl,el}_out_crux/`.

### FairChem/UMA and PACMOF2 in-job

The in-job Parsl/EL demos accept the new workloads (both direct and agent
variants). Install the extras first (`sync_env.sh --extras ...,uma,pacmof2` and,
for PACMOF2, the from-source install — see `scripts/hpc_setup/README.md`).

```bash
# FairChem/UMA thermo screen (GPU)
python scripts/demo/demo_parsl_in_job_direct.py --workload fairchem --device cuda      # Polaris
python scripts/demo/demo_parsl_in_job_direct.py --workload fairchem --device xpu       # Aurora

# PACMOF2 charges on CIFs reachable from the compute node (CPU-only, Crux-friendly)
python scripts/demo/demo_ensemble_launcher_in_job_direct.py \
    --workload pacmof2 --pacmof2-cifs /lus/.../mof1.cif /lus/.../mof2.cif --net-charge 0

# Crux wrapper passthrough
bash scripts/demo/run_crux_demo.sh --workload fairchem
bash scripts/demo/run_crux_demo.sh --workload pacmof2 --pacmof2-cifs /lus/.../mof.cif

# Agent (LLM) variants
python scripts/demo/demo_parsl_in_job_agent.py --workload fairchem --model argo:gpt-4o
python scripts/demo/demo_parsl_in_job_agent.py \
    --workload pacmof2 --pacmof2-cifs /lus/.../mof.cif --model argo:gpt-4o
```

Agent variants on either system require an LLM key and follow the
same pattern as `demo_local_agent.py`.

## Tips

- `--molecules water methane` to run on a subset (faster iteration).
- `--output-dir /custom/path` to redirect CSV + per-molecule JSON.
- The first run on a fresh endpoint / fresh venv will be slow because
  MACE-MP downloads a ~hundred-MB model. Subsequent runs hit the cache
  at `~/.cache/mace/`.

## Known caveats

- **`langchain-mcp-adapters` must be pinned to `0.1.14`** for the
  `*_agent.py` scripts to import. Versions `>=0.2.0` import
  `langchain_core.messages.content` (a 1.x API) which doesn't exist in
  `langchain-core 0.3.x` — and `langgraph 0.4.7` (pinned in
  `pyproject.toml`) constrains us to `langchain-core 0.3.x`. Fix in
  `.cg_env`:
  ```bash
  pip install 'langchain-mcp-adapters==0.1.14'
  ```
  This is an **env-only pin** — `pyproject.toml` still lists
  `langchain-mcp-adapters` unpinned, so a fresh `pip install -e .`
  will regress to `>=0.2`. Re-run the pin command after any clean env
  rebuild. The durable fix (one-line edit to `pyproject.toml`) was
  deferred per user request.
- `ensemble-launcher` is not on PyPI for Python 3.12; the in-job EL
  demos only work on HPC where `scripts/hpc_setup/install_remote.sh`
  builds it from source.

## See also

- `scripts/smoke/` — pass/fail regression validators (trivial payload).
- `scripts/hpc_setup/{README.md,e2e_test_runbook.md}` — install
  ChemGraph + start a Globus Compute endpoint on Polaris/Aurora.
- `scripts/globus_compute_example/` — looser tutorial-style examples,
  predecessors of these demos.
- `src/chemgraph/execution/` — the production backends the demos call.
- `src/chemgraph/mcp/mace_mcp_hpc.py` — the MCP server every agent
  demo spawns as a subprocess.
