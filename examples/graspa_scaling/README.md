# gRASPA Aurora scaling example

This runner uses the original `ChemGraph(workflow_type="graspa_mcp")`
planner/executor/analyst graph. The planner delegates scientific tasks, executor
agents call simulation tools, and the analyst invokes aggregation and ranking
tools before reporting results. Parsl schedules individual simulations.

See the [workflow reference](workflow.md) for schemas, tool behavior, and results.
The PBS launcher targets Aurora Linux and requires Bash 4.3+ and GNU coreutils.
Selected CIFs are referenced by symlinks in `inputs/`; server discovery resolves
the original source paths. `screening.json` saves the selection/settings and
`request.txt` saves the query. A small result interceptor writes MCP records to
JSONL and returns counts and paths so the model need not read thousands of rows.
Tool selection, polling, and analysis remain under the graph's agents.

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
bash examples/graspa_scaling/run.sh --interactive
```

This uses Parsl, the first 20 sorted CIFs from the `coremof_database/databases`
dataset and 10,000 cycles per phase: 40 simulations at the same 298 K and
960/320 Pa conditions. Both modes default to `alcf:openai/gpt-oss-120b`;
set `CG_MODEL` to select another model. It uses `CG_ENV`, the active venv,
or the existing `/lus/flare/projects/ChemGraph/thang/ChemGraph/venv`.
Environment overrides and `CG_SETUP_FILE` are supported as in batch mode.
Unset old workload/output overrides if you want these defaults.

Each invocation defaults to a fresh
`graspa_scaling_runs/interactive-<job>-<time>-<pid>/` directory, printed at startup.
Agent output appears in the terminal and `agent.log`, including live executor
replies, tool starts, batch status/counts, artifact paths, and errors. Entries
include UTC timestamps and worker labels; large payloads are summarized.
Replies appear when each model call finishes, rather than token by token.
Ctrl-C cleans up the
client and MCP server. The launcher uses the current allocation; it does not
submit another PBS job or extend its walltime. The default client timeout is
50 minutes including MCP readiness. Inspect `outcome.json`, `results.csv`, the selected `rankings_<id>.csv`, and
`timing.json` at completion. A successful 20-CIF run has 40 records and selects
four candidates when all structures succeed. Finish or stop the previous MCP run before starting this
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

A live 20-CIF run on Aurora on 2026-09-23 validated the previous deterministic
pipeline, not this restored graph. Repeat a small real-engine run before scaling.
Full-scale and Ensemble Launcher runs remain unverified. The command above
requests 512 nodes; its inherited three-hour walltime is not a runtime guarantee.

Results go to `graspa_scaling_runs/PBS_JOBID/` under the checkout.
`mcp.log` captures service output. `agent.log` captures MCP readiness, the
planner/analyst output, and executor progress while subgraphs are still running.
The Python client waits up to `CG_STARTUP_TIMEOUT` (default 300 seconds) for an
MCP handshake and the required tools. Connection failures are retried during
startup only; workflow failures do not trigger reconnection or resubmission.
There is no separate `readiness.log`.
Saved `state_thread_*.json` snapshots contain completed executor histories in
`executor_logs`; live progress does not wait for these snapshots.
`tool_results/*.jsonl` retain simulation outcomes, including failures.
The analyst aggregates these into `results.csv` and writes its selection to
`rankings_<id>.csv`; tool responses report the actual path. Paired ranking columns
are `input_structure_file`, `uptake_ads`, `uptake_des`, and `working_capacity`.
Each simulation retains its own diagnostics beneath `simulations/`.

`response.txt` saves the final answer. After the graph finishes, `outcome.json`
checks terminal results against the selected structures/conditions; missing or
failed results or an unfinished ranking cause a nonzero exit. This check does
not enforce model-generated parameters before submission. `timing.json` records
client walltime and exit status; a forcibly killed process may not write it.
The original graph does not produce a workflow journal or canonical `analysis.json`.

Optional exported variables (also pass their names with `qsub -v`):
`CG_CIF_DIR`, `CG_LIMIT`, `CG_RUN_DIR`, `N_CYCLES`,
`ADS_TEMP_K`, `DES_TEMP_K`, `ADS_PRESSURE_PA`, `DES_PRESSURE_PA`,
`CG_OMP_NUM_THREADS`, `CHEMGRAPH_PARSL_MAX_WORKERS_PER_NODE`,
`CHEMGRAPH_GRASPA_EXECUTABLE`, `CG_MODEL`, `CG_BASE_URL`,
`CG_SIMULATION_TIMEOUT`, `CG_STARTUP_TIMEOUT`, `CG_WAIT_TIMEOUT`, `CG_AGENT_TIMEOUT`, `CG_RECURSION_LIMIT`.
`CG_STARTUP_TIMEOUT` maps to Python's `--startup-timeout` and bounds the readiness
phase. `CG_AGENT_TIMEOUT` bounds the entire client process, including imports,
readiness, execution, and analysis (10,200 seconds in batch mode).
`CG_WAIT_TIMEOUT` bounds MCP transport reads; `CG_RECURSION_LIMIT` defaults to
100 graph steps. The agent chooses when to call status/result tools.
Changing timeouts does not extend PBS walltime.

For comparisons across node counts, keep nine CIFs per node and both conditions,
so each node receives 18 simulations. `CG_LIMIT` takes a deterministic prefix;
it does not reproduce independent random samples at each scale.

The batch launcher always starts a new server/allocation and requires a fresh
output directory. Pending Parsl futures do not survive server shutdown.
There is no example `--resume` or custom request contract. Use existing server
job tools to inspect accepted batches; do not treat a new PBS submission as
recovery of unfinished work. The server stores its tracker at
`~/.chemgraph/graspa_jobs.json`; avoid independent servers writing the same tracker.
