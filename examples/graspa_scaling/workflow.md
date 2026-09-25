# gRASPA example workflow reference

Companion reference for the [Aurora scaling example](README.md).

ChemGraph supports the SYCL gRASPA engine with its bundled H2O model. CUDA,
CO2, and N2 simulations are not supported by this integration. The force-field
parameters and cycle definitions are unchanged by the runtime cleanup.

Install ChemGraph, then configure the executable **on each execution worker**:

```bash
export CHEMGRAPH_GRASPA_EXECUTABLE=/path/to/graspa-sycl/bin/sycl.out
export CHEMGRAPH_LOG_DIR=/path/to/writable/results
```

Without the executable override, ChemGraph searches for `sycl.out` on `PATH`.
An invalid explicit override fails instead of selecting another binary. Load
the SYCL runtime/modules required by your installation before starting workers.
`OMP_NUM_THREADS=1` and `ZE_FLAT_DEVICE_HIERARCHY=FLAT` are defaults; existing
environment values are respected. Installing ChemGraph does not install gRASPA.

## Direct calculation

```python
from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools.graspa_core import run_graspa_core

result = run_graspa_core(graspa_input_schema(
    input_structure_file="/path/to/framework.cif",
    adsorbate="H2O",
    temperature=298.15,  # K
    pressure=1000,       # Pa, not relative humidity
    n_cycles=10000,      # each initialization and production phase
    output_directory="water-screening",
    timeout_seconds=None,  # optional positive wall-time limit
))
print(result)
```

The LangChain `run_graspa` tool accepts the same schema under `graspa_input`
and retains its successful float return (mol/kg). It raises an actionable error
on failure. The core function returns a result dictionary; invalid input paths
or schemas fail before execution. A prepared run retains its diagnostics even
when process startup, execution, or parsing fails.

## Artifacts and migration

Every invocation creates a new run directory. Relative output roots resolve
under `CHEMGRAPH_LOG_DIR`, or the worker's current directory when unset. The
default root is `graspa_runs`. The input CIF is never modified.

`output_result_file` is the stdout filename within that unique directory,
defaulting to `raspa.log`. Legacy directory-qualified values still select the
parent output root, with a warning; migrate to `output_directory` plus a bare
filename. Do not combine both root specifications. Names reserved for the CIF,
templates, stderr, or JSON metadata are rejected.

Use the returned `run_dir`, `stdout_path`, `stderr_path`, and `results_path`;
do not reconstruct paths from temperature or pressure. `input_structure_file`
identifies the original source, while `cif_path` identifies its run-local copy.
`results.json` includes run ID, conditions, exit code, elapsed seconds, and
uptake in mol/kg. Failures have `status="failure"`, null uptake, and
`error_type`/`message`. A nonzero exit can never be accepted as a successful
calculation, even when partial stdout contains an uptake value.

## MCP ensembles

The maintained server is `chemgraph.mcp.graspa_mcp_hpc`. Configure its execution
backend explicitly; backend initialization is lazy. For workers sharing the
server filesystem:

```bash
export CHEMGRAPH_EXECUTION_BACKEND=parsl  # or local, ensemble_launcher
export COMPUTE_SYSTEM=aurora             # select your backend's configuration
python -m chemgraph.mcp.graspa_mcp_hpc --transport stdio
```

For HTTP clients, use `--transport streamable_http --host 127.0.0.1 --port 9001`;
the endpoint is `http://127.0.0.1:9001/mcp/`. The local backend can also run gRASPA
when its workers have the required executable and SYCL runtime. Each worker
needs the same ChemGraph version, readable CIFs, and a writable output root.
Set `CHEMGRAPH_GRASPA_EXECUTABLE`, runtime modules, and `CHEMGRAPH_LOG_DIR` in the
worker environment; setting them only on a remote client does not configure
the workers.

Call `run_graspa_ensemble` with this MCP argument object:

```json
{
  "params": {
    "input_structures": ["/shared/cifs/framework.CIF"],
    "adsorbate": "H2O",
    "conditions": [
      {"temperature": 298.15, "pressure": 1000},
      {"temperature": 298.15, "pressure": 2000}
    ],
    "output_directory": "water-screening",
    "output_result_file": "raspa.log",
    "timeout_seconds": null
  }
}
```

`input_structures` accepts a directory or an explicit file list. Directory
scans select regular CIF files, case-insensitively. Explicit lists preserve
their order and duplicates; relative listed filenames can resolve under
`CHEMGRAPH_LOG_DIR`. Every structure/condition pair gets an independent job ID
and run directory. One structure and one condition run a single simulation.
Invalid local requests are rejected before submission; files that disappear
after discovery become per-job failures.

### Remote workers without a shared filesystem

For Globus Compute, set `CHEMGRAPH_EXECUTION_BACKEND=globus_compute` and configure
the endpoint through `GLOBUS_COMPUTE_ENDPOINT_ID`. Pre-stage the CIFs on the
worker filesystem. Optional `transfer_files`, `check_transfer_status`, and
`list_remote_files` tools are registered at server startup when Globus Transfer
is configured; wait for staging to finish before requesting discovery.

Local input mode is rejected on backends without a shared filesystem. Replace
`input_structures` with `remote_structure_directory`:

```json
{
  "params": {
    "remote_structure_directory": "/remote/project/staged-cifs",
    "adsorbate": "H2O",
    "conditions": [{"temperature": 298.15, "pressure": 1000}],
    "output_directory": "/remote/project/water-screening",
    "discovery_timeout_seconds": 300,
    "timeout_seconds": null
  }
}
```

The worker resolves the remote directory and discovers all regular CIFs in it.
Discovered POSIX and Windows absolute paths are preserved regardless of the
MCP server's operating system.
No implicit file transfer or inline-CIF transport is performed. Discovery is
awaited without blocking the MCP event loop. Its timeout defaults to 30 seconds,
includes submission, queueing, and discovery execution, and can be disabled with
`null`. A timeout or cancelled request stops waiting but cannot forcibly stop a
blocking SDK submission or an already-running discovery probe. Late discovery
results do not start simulations. Discovery failure, including a probe cancelled
by the backend, returns a tool error without closing the MCP connection and
submits no simulations; check the directory/backend or increase the timeout.

`timeout_seconds` independently limits each simulation process after startup.
It is unset by default and excludes backend queue time. Output roots and legacy
directory-qualified stdout names are passed unchanged to workers for resolution.

### Results, polling, and cancellation

Synchronous backends return `{"status": "completed", "results": [...]}`.
Async backends return `{"status": "submitted", "batch_id": "...", ...}`.
Use `check_job_status(batch_id)` to poll and `get_job_results(batch_id)` to
retrieve records. `get_job_results(batch_id, include_partial=true)` returns
finished records while other tasks remain pending. `list_jobs()` lists batches.
Polling reports terminal `completed`, `partial`, or `failed` states; a terminal
batch can contain failed simulations. Inspect each record's `status` and error.

Every record retains job ID, source structure, adsorbate, and conditions, even
when submission or execution fails. Prepared runs also return the artifact
paths described above. Uptake is in mol/kg; failed uptake is null. `raspa.log`
is plain engine stdout, while `results.json` is a parsed result record.

`cancel_job(batch_id)` attempts to cancel pending futures; it does not promise
to stop already-running processes. The server persists tracked batches under
`~/.chemgraph/graspa_jobs.json`. Cached results and Globus task IDs support
restart recovery. Unfinished ordinary futures without remote task IDs cannot
be recovered after a restart.

### Deprecated Parsl entry point

`chemgraph.mcp.graspa_mcp_parsl` remains available for existing clients and
initializes Parsl lazily. Its `run_graspa_parsl_app` callable still returns a
future. Its ensemble tool waits for all results and returns text containing an
absolute `simulation_results.jsonl` path. Each batch writes a fresh summary
directory under the worker-resolved output root, including failed records.
The native graph accepts this response when the JSONL file is readable on the
client. New clients should use the maintained server, whose record responses
also support workers without a shared filesystem.

## Planner/executor/analyst graph

`ChemGraph(workflow_type="graspa_mcp")` uses the original LangGraph agent flow:

1. The planner reads conversation history and creates tasks by scientific intent.
2. `Send` dispatches executor subgraphs. Each executor chooses from the supplied
   tools in an agent/tool loop. Submitted batches are polled and collected through
   the job tools before the executor reports completion.
3. Executor summaries and logs join before the planner runs again. It may delegate
   more work or select the insight analyst.
4. The analyst calls aggregation and ranking tools through its own `ToolNode`
   loop, then returns the final answer. It can also analyze existing results
   without submitting simulations. `FINISH` ends a request directly.

One task can contain a directory and both adsorption/desorption conditions.
Parsl schedules individual simulations; there is no need for one model task per CIF.
The graph preserves request context alongside each assigned task.

Supply simulation tools through `ChemGraph(tools=...)` and analysis tools through
`data_tools`. The [scaling runner](run_graspa.py) demonstrates this with the
maintained gRASPA MCP server and LangChain wrappers around
`aggregate_simulation_results` and `rank_mofs_performance`.
`PromptConfig.planner`, `.executor`, and `.aggregator` override the respective
agent prompts. Planning uses `PlannerResponse` with `next_step`,
`thought_process`, and optional tasks, rather than a frozen workflow contract.
`return_option="state"` exposes messages, executor summaries, and executor logs;
`last_message` returns the final agent message. A completed graph is not by itself
proof that all scientific calculations succeeded: inspect tool outcomes.

### Live progress

The scaling example attaches an example-local callback through `agent.run()`.
It prints timestamped model starts, executor replies, tool starts/results, and
errors into `agent.log`, with worker labels for concurrent executors. These
events are visible before executor subgraphs finish. Existing planner/analyst
response printing is preserved without replaying those replies in the callback.
Model replies are printed after each call completes, not token by token.

Tool output is bounded to summaries of settings, status/counts, and artifact
paths; bulk simulation records remain in `tool_results/*.jsonl`. Completed
executor history also remains in `executor_logs` in saved state snapshots.
The callback only observes execution: it adds no submissions, model calls, or
polling, and logging failures do not stop the workflow.

### Result artifacts and numerical analysis

The example uses an MCP result interceptor that writes returned simulation
records to `tool_results/*.jsonl` and replaces large tool payloads with status,
counts, and actual file paths. It does not submit jobs, poll, retry, or rank.
Both the visible tool response and its structured artifact remain compact.
The analyst's local tools can read these files on the client filesystem.

The analyst calls `aggregate_simulation_results` to produce `results.csv`, then
`rank_mofs_performance` to write a selected `rankings_<id>.csv`. Numerical work
remains deterministic inside these tools. The original request supplies the
conditions and selection rule, including `top_percentile` or `min_cutoff`.

Ranking uses exact temperature/pressure matches and full original source paths.
Failed repeats exclude the corresponding structure. Failed, mock, negative,
and nonfinite uptake is never treated as zero. Successful repeats are averaged
before calculating adsorption minus desorption uptake, in mol/kg. Fractional
selection uses `ceil(fraction * valid_candidates)`. Paired rankings contain
`input_structure_file`, `uptake_ads`, `uptake_des`, and `working_capacity`;
single-condition rankings contain `input_structure_file` and `absolute_uptake`.

### Lifecycle and validation

The graph uses ordinary LangGraph state and checkpoints. There is no custom
workflow journal, request contract, `graspa_options`, or example `--resume`.
The server's existing job tracker remains available for inspecting accepted
batches. Pending ordinary Parsl futures require the original server/allocation
to stay alive. A new PBS job does not recover unfinished work.

The example requires a fresh output directory, saves the selected workload in
`screening.json`, and checks terminal records against the requested source and
condition multiplicities after the graph finishes. Missing or failed results,
or an unfinished ranking tool, produce a nonzero exit and `outcome.json`.
This post-run check does not enforce model-generated parameters before submission
or replace the analyst's numerical work. Use `--recursion-limit` for the graph
and `--wait-timeout` for MCP transport reads. `--startup-timeout` (default 300
seconds, forwarded from `CG_STARTUP_TIMEOUT`) bounds the initial MCP handshake
and required-tool check. Readiness progress goes to `agent.log`; connection
failures are retried only before the session is ready. Missing tools fail
immediately, and workflow failures are never retried by the readiness helper.
The launcher bounds total client walltime, including readiness, with
`CG_AGENT_TIMEOUT`.

A 20-CIF Aurora run on 2026-09-23 validated the superseded deterministic graph
with Parsl and ALCF `openai/gpt-oss-120b`. That evidence does not validate this
restored agent flow. Run a fresh small real-engine validation before scaling;
full-scale and Ensemble Launcher validation remain outstanding. Hermetic tests
exercise graph routing, tool handoff, numerical analysis, and failure reporting
without live LLMs, GPUs, or external endpoints.
