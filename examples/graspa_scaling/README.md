# Screen your CIF database with gRASPA

Use ChemGraph and Parsl on Aurora to run and rank **H2O, CO2, or N2** adsorption.
Each run uses one gas and two temperature/pressure conditions per CIF.

## Set your paths

From your ChemGraph checkout, replace these with your own absolute paths:

```bash
export CG_CIF_DIR=/path/to/your/cif_database
export CG_ENV=/path/to/your/chemgraph_venv
export CHEMGRAPH_GRASPA_EXECUTABLE=/path/to/your/graspa/sycl.out
```

The CIF directory must be accessible to the workers and contain `.cif` files
directly; subdirectories are not searched. The venv needs ChemGraph's dependencies
and Parsl. Export a valid `ALCF_ACCESS_TOKEN` using your normal authentication
workflow. The default model is `alcf:openai/gpt-oss-120b`; override with `CG_MODEL`.

Preview all inputs without services, model calls, or output files:

```bash
python3 examples/graspa_scaling/run_graspa.py --dry-run \
  --input-dir "$CG_CIF_DIR" --limit 0 --adsorbate H2O --output-dir /tmp/graspa-preview
```

## Run in an existing Aurora allocation

```bash
# First 20 CIFs (40 simulations), using H2O:
CG_LIMIT=20 bash examples/graspa_scaling/run.sh --interactive

# All CIFs in your database, using H2O:
CG_LIMIT=0 bash examples/graspa_scaling/run.sh --interactive

# Select another gas; CO2 or N2 are accepted:
CG_ADSORBATE=CO2 CG_LIMIT=0 bash examples/graspa_scaling/run.sh --interactive
```

**`CG_LIMIT=0` means all CIFs**; a positive value selects the first N sorted files.
`CG_ADSORBATE` defaults to `H2O` and maps to Python's `--adsorbate` option.
Use separate runs for different gases; mixture simulations are not supported.

| Setting | Interactive default | Batch default |
| --- | --- | --- |
| `CG_LIMIT` | 20 CIFs | 0 (all CIFs) |
| `N_CYCLES` | 10,000 | 2,000,000 |
| `CG_WAIT_TIMEOUT` | 2,700 seconds | 9,900 seconds |
| `CG_AGENT_TIMEOUT` | 3,000 seconds | 10,200 seconds |

Cycles apply **each to initialization and production**. All gases inherit the
H2O example conditions unless overridden: 298 K and 960/320 Pa. Set
`ADS_TEMP_K`, `DES_TEMP_K`, `ADS_PRESSURE_PA`, and `DES_PRESSURE_PA` for your
scientific workload. Selecting a gas or `CG_LIMIT=0` does not change these settings.

`CG_AGENT_TIMEOUT` includes startup, simulations, and analysis; it does not
extend PBS walltime. `CG_STARTUP_TIMEOUT` defaults to 300 seconds. Interactive
mode prints live progress, uses a fresh output directory, and cleans up on Ctrl-C.
It uses your current allocation and does not submit another PBS job.

## Submit through PBS

Pass the exported paths to your job. For a small run:

```bash
qsub -A YOUR_PROJECT -q debug -l select=1,walltime=01:00:00 \
  -v CG_ENV,CG_CIF_DIR,CHEMGRAPH_GRASPA_EXECUTABLE,ALCF_ACCESS_TOKEN,CG_ADSORBATE=H2O,CG_LIMIT=20,N_CYCLES=10000,CG_WAIT_TIMEOUT=2700,CG_AGENT_TIMEOUT=3000 \
  examples/graspa_scaling/sub.graspa.aurora
```

Use `CG_LIMIT=0` for all CIFs and select nodes, walltime, conditions, and cycles
for your workload. Pass additional exported overrides by name in `qsub -v`.
Without resource overrides, the PBS file requests **512 nodes, three hours,
queue `prod`, account `IQC`**. This launcher targets Aurora Linux, not other clusters.

## Results

The launcher prints the output directory under `graspa_scaling_runs/`.
Override `CG_RUN_DIR` with an unused directory if needed.

- `agent.log`: readiness, planner/analyst output, and live executor/tool progress.
- `mcp.log` and `simulations/`: server and individual simulation diagnostics.
- `results.csv` and `rankings_<id>.csv`: aggregated results and selected candidates.
- `outcome.json`: completion check, including the requested gas; failures cause a
  nonzero exit. `response.txt` contains the final report.
- `screening.json`, `tool_results/*.jsonl`, and `state_thread_*.json`: input
  settings, raw records, and agent history.

Use one server at a time: the job tracker is shared, and new runs do not resume
unfinished work. See the [workflow reference](workflow.md) for templates, schemas,
ranking, and recovery limits, or run `bash examples/graspa_scaling/run.sh --help`.
