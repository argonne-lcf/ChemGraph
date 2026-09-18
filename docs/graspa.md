# gRASPA-SYCL: H2O adsorption

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

## Native ChemGraph workflow

`ChemGraph(workflow_type="graspa_mcp")` plans logical ensembles, prepares their
validated requests, then submits, polls, and collects in Python. All tasks join
before Python analysis runs; the final LLM call explains the saved analysis.
One directory with adsorption and desorption conditions needs one ensemble,
not one model call per CIF. File lists and complete records remain in artifacts;
model messages and checkpoints contain counts, paths, and at most five preview
rows. Optional `data_tools` can supplement the explanation.

```python
from chemgraph.agent.llm_agent import ChemGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

client = MultiServerMCPClient({"graspa": {
    "transport": "streamable_http", "url": "http://127.0.0.1:9001/mcp/",
}})
async with client.session("graspa") as session:
    agent = ChemGraph(
        workflow_type="graspa_mcp",
        tools=await load_mcp_tools(session),
        graspa_options={
            "run_directory": "/shared/screening/run-001",
            "poll_interval_seconds": 15,
            "wait_timeout_seconds": 3600,
            "resume": False,
        },
        return_option="state",
        enable_memory=False,
    )
    state = await agent.run(
        "Screen H2O on all CIFs in /shared/cifs at 298 K, adsorption 960 Pa "
        "and desorption 320 Pa, with 2000000 cycles per phase. Use one ensemble "
        "and rank the top 20% by adsorption minus desorption uptake."
    )
    print(state["workflow_status"], state["analysis"])
```

The asynchronous snippet belongs inside your application coroutine. Configure
`model_name` and provider authentication as for other ChemGraph workflows.
`return_option="last_message"` returns the explanation message instead;
`workflow_finished` events carry the scientific status with either return option.
Statuses are `completed`, `partial`, `failed`, or `incomplete`. An incomplete
collection never generates a new ranking. A report-generation error is saved
in `report_error.json` and does not erase the collected scientific outcome.

`graspa_options` accepts only the four fields above. Intervals and timeouts must
be positive finite seconds. The collection timeout applies per logical task,
including submission, queueing, and polling; it does not stop remote work or
replace each simulation's `timeout_seconds`. With no explicit run directory,
the graph creates a unique `graspa_workflows/<id>` beneath `log_dir` (or
`CHEMGRAPH_LOG_DIR`). Relative explicit run directories resolve against
`CHEMGRAPH_LOG_DIR`. A populated workflow directory requires `resume=True`.
`config={"max_concurrency": 4}` on `agent.run` bounds model preparation and
concurrent ensemble collection; it does not configure Parsl workers.

Use `PromptConfig.planner`, `.executor`, and `.aggregator` to override planning,
request preparation, and explanation respectively. Planning must return a
`GraspaPlan` and preparation a `graspa_input_schema_ensemble`; old prompts that
route between agents or perform model-driven polling are incompatible. The
workflow always owns the join and canonical numerical analysis.

### Artifacts and ranking rules

- `workflow.json`: query, frozen plan and requests, submission intent, accepted
  batch IDs, progress, collection errors, and task artifact paths.
- `plan.json`, `task_<index>.jsonl`: the plan and collected per-task records.
- `results.jsonl`, `results.csv`: all collected outcomes, including failures and
  original source identities; `analysis.json`: bounded summary and conditions.
- `rankings.csv`, `top_candidates.csv`, `excluded.json`: full valid ranking,
  selected candidates, and exclusions when ranking is requested and collection
  is complete. `response.txt` contains the final explanation.

Ranking uses exact temperature/pressure matches (298 K does not match 298.15 K)
and full source paths, so equal CIF basenames in different directories stay
separate. All requested repeats at each ranking condition must succeed. Failed,
mock, negative, or nonfinite uptake is never treated as zero. Legacy records
with missing conditions exclude their source from ranking because they could
be unidentified failed repeats. Successful repeats are averaged before
computing adsorption minus desorption uptake, in mol/kg. The selected count is
`ceil(top_fraction * valid_candidates)`; the fraction must be in `(0, 1]`.

The analysis MCP tools `aggregate_simulation_results` and
`rank_mofs_performance` share these numerical rules, retain JSONL input support,
and preserve failure records and full source paths. They are optional for the
native graph, which performs canonical analysis locally.

### Recovery limits

To resume, use the original query, the same `run_directory`, and `resume=True`.
The graph loads its frozen requests without calling the planner/preparer again,
reuses accepted batch IDs, and validates saved records. It never automatically
resubmits an accepted batch. Concurrent writers to one run directory are rejected.
A timeout or cancelled client can leave remote simulations running.

A lost or cancelled submission acknowledgment leaves `phase="submission_unknown"`
(or crash-time `submitting`) in `workflow.json`. Resume stops that task and marks
collection incomplete. Reconcile against the original server's `list_jobs()`
and saved request before editing the journal: a confirmed accepted batch needs
`phase="submitted"`, its `batch_id`, and positive `n_tasks`; only reset to
`unsubmitted` after confirming no work was accepted. Back up the journal first.
There is no automatic reconciliation or exactly-once guarantee across network
failures. Ordinary Parsl futures require the original server/allocation to stay
alive; restarting PBS is not recovery of unfinished work. Legacy JSONL summaries
must remain readable on the client. Do not change the frozen scientific request
while resuming.

The self-contained [Aurora runner](../scripts/graspa_scaling/README.md) uses this
native graph with the 4,608-CIF reference workload and supports a four-CIF smoke.

## Validation status

Hermetic tests validate preparation, parsing, isolation, and failure handling
without downloading models or requiring a GPU. Before using this integration
for scientific results, run a small H2O calculation on your SYCL installation,
compare the parsed uptake with stdout, and record the executable version and
sanitized output. Real-engine validation is a collaborator handoff requirement.
