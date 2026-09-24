# gRASPA Aurora scaling example

This runner uses `ChemGraph(workflow_type="graspa_mcp")`: the LLM plans one
ensemble with both conditions, Python submits and collects all records, Python
ranks working capacity, and the LLM explains the saved summary. Counts, paths,
and up to five preview rows enter the reporting context. The scripts are
self-contained and do not depend on local demo files.

See the [workflow reference](workflow.md) for request schemas, MCP behavior,
output fields, and recovery details.

The PBS shell launcher targets Aurora Linux and requires Bash 4.3+ and GNU
coreutils (`timeout` and `tail --pid`).

The selected CIFs are referenced by symlinks in the run's `inputs/` directory;
shared-file discovery resolves these links to the original full source paths.
The model receives a directory reference, even for a 4,608-CIF run. Source CIFs
are never copied or modified. `screening.json` saves the original selection and
settings, and `request.txt` saves the query used for resume.
The runner also passes those settings and the selected source paths as a typed
request contract. The native workflow checks the complete workload before
submission, including both conditions, cycles, output root, and ranking settings.
Planning/preparation get three attempts to correct invalid output; failures are
recorded in `validation-<stage>-<id>.json` without storing raw model responses.

## Reference workload

- 4,608 CIFs from the supplied `512_nodes/cif_files` directory.
- H2O, 298 K: adsorption 960 Pa (30% × 3200), desorption 320 Pa (10% × 3200).
- 2,000,000 cycles **each** for initialization and production.
- 9,216 simulations total; nine Parsl workers/node, or 4,608 slots at 512 nodes.
- One ChemGraph agent controls submission. Agent concurrency does not set
  the number of Parsl workers.
- `OMP_NUM_THREADS=1` is applied after module setup on client and workers.
- PBS defaults match the reference: 512 nodes, `prod`, three hours, account
  `IQC`. Override the account/queue/resources in `qsub` for your allocation.

## Prepare and inspect

From the ChemGraph checkout:

```bash
cd /lus/flare/projects/ChemGraph/thang/ChemGraph/.worktrees/pr244-validation
export CG_ENV=/lus/flare/projects/ChemGraph/thang/ChemGraph/venv
python examples/graspa_scaling/run_graspa.py --dry-run --output-dir /tmp/graspa-preview
```

The dry run uses only the Python standard library, checks the input paths,
and prints counts/conditions without starting MCP, calling an LLM, or writing
outputs. It does not validate the SYCL executable or scientific convergence.

The shell launcher and Python CLI default to `alcf:openai/gpt-oss-120b`;
override with `CG_MODEL` in the launcher or `--model` in Python. No inference URL is forced:
the model loader selects it unless `CG_BASE_URL` or `--base-url` is configured.
For an `alcf:*` model, use your existing authentication procedure to populate
`ALCF_ACCESS_TOKEN`. To refresh credentials at job start,
use an absolute shared `CG_SETUP_FILE` that loads the modules, activates the
environment, and obtains the token, then pass its name instead of `CG_ENV`
and `ALCF_ACCESS_TOKEN`. It is also sourced by Parsl workers, so keep setup
repeatable. No credentials are stored in these scripts.

## Interactive 20-CIF run

Inside an existing Aurora PBS interactive allocation, with a valid
`ALCF_ACCESS_TOKEN` exported in that shell:

```bash
cd /lus/flare/projects/ChemGraph/thang/ChemGraph/.worktrees/pr244-validation
CG_MODEL=alcf:openai/gpt-oss-120b bash examples/graspa_scaling/run.sh --interactive
```

This uses Parsl, the first 20 sorted CIFs from the `coremof_database/databases`
dataset and 10,000 cycles per phase: 40 simulations at the same 298 K and
960/320 Pa conditions. The command explicitly selects the validated
`alcf:openai/gpt-oss-120b` model; without `CG_MODEL`, interactive mode defaults
to `alcf:nemotron-3-ultra`. It uses `CG_ENV`, the active venv,
or the existing `/lus/flare/projects/ChemGraph/thang/ChemGraph/venv`.
Environment overrides and `CG_SETUP_FILE` are supported as in batch mode.
Unset old workload/output overrides if you want these defaults.

Each invocation defaults to a fresh
`graspa_scaling_runs/interactive-<job>-<time>-<pid>/` directory, printed at startup.
Agent output appears in the terminal and `agent.log`. Ctrl-C cleans up the
client and MCP server. The launcher uses the current allocation; it does not
submit another PBS job or extend its walltime. The default client timeout is
50 minutes. Inspect `analysis.json` and `timing.json` at completion; a fully
successful run has 40 records, 20 valid structures, four selected candidates,
and exit status zero. Finish or stop the previous MCP run before starting this
one, since the job tracker is shared.

## Small allocation first

This selects the first four sorted CIFs from the reference dataset; it is a
wiring and runtime test, not a convergence study.

```bash
qsub -q debug -l select=1,walltime=01:00:00 \
  -v CG_ENV,ALCF_ACCESS_TOKEN,CG_LIMIT=4,N_CYCLES=10000,CG_WAIT_TIMEOUT=2700,CG_AGENT_TIMEOUT=3000 \
  examples/graspa_scaling/sub.graspa.aurora
```

After checking successful records and comparing parsed uptake with each
`stdout_path`, test more nodes or the full cycle count before scaling up.
Account `IQC` is inherited from the script; use `-A PROJECT` if needed.

## Full reference run

```bash
qsub -v CG_ENV,ALCF_ACCESS_TOKEN examples/graspa_scaling/sub.graspa.aurora
```

A live 20-CIF run on Aurora with Parsl and `alcf:openai/gpt-oss-120b` completed
on 2026-09-23: 40 successful records, 20 ranked structures, and four selected
candidates. This validates the small workflow; full-scale and Ensemble Launcher
runs remain unverified. The command above uses 512 nodes when scheduled. The
three-hour walltime is inherited from the reference and is not a runtime guarantee.

Results go to `graspa_scaling_runs/PBS_JOBID/` under the checkout.
`mcp.log`, `readiness.log`, and `agent.log` capture service startup and the
workflow. `workflow.json` retains the plan, frozen requests, submission intent,
accepted IDs, and status; `task_1.jsonl`, `results.jsonl`, and `results.csv` retain all
collected outcomes.
`rankings.csv` contains complete successful pairs, and
`top_candidates.csv` contains the first `ceil(0.2 × successful pairs)` rows.
Both contain `input_structure_file`, `uptake_ads`, `uptake_des`, and
`working_capacity`; the `analysis.json` preview uses the same fields. Raw
per-simulation records retain their diagnostics and artifact paths.
Failed repeats exclude their structure from ranking. Partial, failed, and
incomplete workflows cause a nonzero client exit. `excluded.json` records
exclusions and `analysis.json` records conditions, counts, and paths.
`timing.json` measures client walltime, excluding MCP startup. A forcibly killed process may not write timing.
Each simulation has its own directory beneath `simulations/`.

Optional exported variables (also pass their names with `qsub -v`):
`CG_CIF_DIR`, `CG_LIMIT`, `CG_RUN_DIR`, `N_CYCLES`,
`ADS_TEMP_K`, `DES_TEMP_K`, `ADS_PRESSURE_PA`, `DES_PRESSURE_PA`,
`CG_OMP_NUM_THREADS`, `CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE`,
`CHEMGRAPH_GRASPA_EXECUTABLE`, `CG_MODEL`, `CG_BASE_URL`,
`CG_SIMULATION_TIMEOUT`, `CG_WAIT_TIMEOUT`, `CG_AGENT_TIMEOUT`.
Changing timeouts does not extend PBS walltime.

For comparisons across node counts, keep nine CIFs per node and both conditions,
so each node receives 18 simulations. `CG_LIMIT` takes a deterministic prefix;
it does not reproduce independent random samples at each scale.

The batch launcher always starts a new server/allocation and requires a fresh
output directory. Pending Parsl futures do not survive server shutdown.
The client has `--resume` for reconnecting to an original, still-running MCP
server with the same output directory and simulation settings; it loads the
saved source list and query without rediscovery. Accepted batches are not
resubmitted. Resume also requires the original request contract; use a fresh
run directory for attempts created before the runner supplied a contract.
Unknown submission acknowledgments require manual reconciliation;
do not treat a new PBS submission as recovery of unfinished work.
The MCP server stores its job tracker at `~/.chemgraph/graspa_jobs.json`; avoid
concurrent independent MCP servers writing that shared tracker during validation.
