# Remote HPC batch jobs with IRI

Deep Agent can stage selected files using Globus Transfer, submit PBS jobs through
IRI, and inspect or retrieve results. The agent can run on a POSIX laptop or server; SSH, local
PBS commands, an MCP server and a persistent execution daemon are unnecessary.
Accepted PBS jobs continue when ChemGraph exits.

This workflow is independent of execution backends and their MCP batch IDs.
For direct qsub on a submission host, use [PBS jobs with skills](pbs_jobs_with_skills.md).

## Configure a target

Use actual collection IDs, allocation and paths for your deployment. The local
run directory **and its `.hpc-inputs` snapshot** must be accessible to the local
Globus collection, with Globus Connect Personal or a managed collection online.
Compute and collection paths may use different prefixes:

```toml
[hpc.targets.polaris]
compute_resource = "polaris"
storage_resource = "eagle"
local_collection = "YOUR_LOCAL_COLLECTION_UUID"
remote_collection = "YOUR_EAGLE_COLLECTION_UUID"
local_root = "/absolute/local/runs"
local_collection_root = "/absolute/collection/runs"
remote_root = "/eagle/YOUR_PROJECT/chemgraph"
remote_collection_root = "/YOUR_PROJECT/chemgraph"
project = "YOUR_PROJECT"
queue = "debug"
setup_script = "/absolute/compute/environment.sh"
python_executable = "/absolute/compute/environment/bin/python"

[hpc.targets.polaris.resources]
node_count = 1
duration = 300
filesystems = "home:eagle"
```

All roots are absolute. `local_root` maps to `local_collection_root`, and
`remote_root` maps to `remote_collection_root`. Resources are resolved and saved
when staging starts. Environment paths are guidance for preparing the launch
script; no setup script or software installer runs automatically.

Authenticate IRI using the existing ALCF authentication provider/cache. Authenticate
Globus Transfer explicitly in a terminal before invoking tools:

```bash
python -m chemgraph.execution.globus_transfer
```

For managed collections requiring additional consent, repeat
`--collection COLLECTION_UUID` for each such collection. Tool calls never prompt
for authorization codes on stdin. Tokens remain in the authentication cache.
Expired access tokens refresh automatically while the refresh authorization is
valid; authorization failures require another explicit login.

```bash
chemgraph run --interactive --workflow deep_agent --config /path/to/selected.toml \
  --deepagent-workspace /absolute/local/runs
```

> Read hpc-batch and the Polaris site reference. Inspect configured targets.
> Prepare an ASE/EMT optimization in a fresh run directory, using relative input
> and output paths and the configured compute Python. Stage its files, wait for
> transfer success, submit once, and report the saved run directory and job ID.

Normal file, transfer, submission and cancellation reviews apply. `--tool` still
restricts the catalog; include every tool the workflow needs or omit it to use
the configured catalog. Never enable `ALCF_IRI_ALLOW_UNSAFE` for this workflow.

## Python interface

```python
from chemgraph.tools.hpc import HPCConfig
from chemgraph.tools.hpc.tools import create_hpc_registry
from chemgraph.graphs.deep_agent import construct_deep_agent_graph

# parsed_config is the TOML configuration already selected by the caller.
config = HPCConfig.model_validate(parsed_config["hpc"])
registry = create_hpc_registry(config)  # names=[] explicitly disables discovery
agent = construct_deep_agent_graph(llm, tool_registry=registry, backend=backend)
```

`initialize_agent(..., workflow_type="deep_agent", hpc_config=config)` also binds
the catalog. Pass either hpc_config or an explicitly configured registry. Tools
are bound per agent; no module-global target configuration is used.

## Run lifecycle

Prepare all scripts and inputs before staging. Use one local directory per
submission attempt. `hpc_transfer_files(direction="stage")` copies the selected
files into `.hpc-inputs`, records SHA-256 identities, and transfers that snapshot
without flattening paths. Local inputs and the snapshot are checked again before
submission. A changed file requires a fresh run and transfer.

The returned `remote_directory` is unique. Use run-relative paths in ASE JSON
for structures, local model weights and outputs; the packaged runner and launch
template set the working directory and `CHEMGRAPH_LOG_DIR` on compute nodes.
Submission resources come from the typed request/target defaults, not #PBS
comments. This first interface exposes node count, walltime, queue, project and
filesystems; MPI/GPU launch commands belong in the reviewed launch script.

The run retains:

- `run.json`: versioned target snapshot, paths, input hashes, mapping and transfer ID.
- `submission.started`: durable exclusive evidence that submission may have begun.
- `submission.json`: exact request, acceptance/uncertainty, IDs and redacted errors.
- `job-status.json`: latest scheduler observation, separate from scientific success.

Submission is serialized per local run. Repeating an unchanged accepted request
returns the saved job ID. A timeout, crash, malformed response or missing scheduler
record never triggers automatic resubmission. `hpc_job_status` reconciles a saved
operation or unambiguous matching active/historical job. Incomplete or ambiguous
searches stay unknown. Keep all evidence, even after errors.

IRI credentials and the submission payload are prepared before recording a
submission attempt. An `authentication_required` response from `hpc_submit_job`
means no scheduler request was sent: authenticate with `alcf_auth`, then retry
the same run directory. Credentials stay in memory and are never saved in run
evidence. Existing runs marked `submission_unknown` still require reconciliation;
authentication does not clear their submission markers.

Restart ChemGraph and ask it to inspect the same run directory. The saved target
snapshot is authoritative, even if current target configuration changed. Do not
move run directories; their local collection mappings are anchored to the original
location. Local locks do not coordinate independent controllers or prevent users
from changing remote files outside these tools.

Diagnostic jobs use separate directories from production runs. Keep their output
as evidence when reusing verified environments. No software is installed automatically.

## Inspection and results

IRI file reads are bounded, and filesystem operations may return `task_id`.
Pass that ID back as `operation_id` to the same inspection tool. The response
retains task status and result, or explicitly identifies truncation. Narrow large
listings or retrieve files with Globus. The public ALCF guide currently describes
restricted filesystem access; a permission error is not a missing file.

Retrieve explicitly named remote files with `hpc_transfer_files(direction="retrieve")`.
They arrive under `run_dir/retrieved`; poll the returned transfer ID. Existing or
pending destinations require an explicit overwrite request. No result retrieval
implicitly submits a calculation.

ASE uses the existing schema and core engine. The packaged runner exits 0 for
convergence, 2 for nonconvergence, and 1 for failure. Inspect scientific fields,
stdout/stderr and artifacts. Early failures may leave no result JSON. Cancellation
requires the run's full accepted scheduler ID; acknowledgment does not prove the
job has stopped.

## Verification and deployment status

Hermetic tests exercise transport contracts, retries, token refresh, concurrent
submission, uncertain acceptance, restart recovery, reviews and real ASE/EMT
in a separate simulated compute directory. Live Polaris execution is a deployment
acceptance step, not part of the default test suite.

The adapter uses the documented ALCF endpoints and the public IRI reference
JobSpec/Job/task models. The deployed OpenAPI endpoint was inaccessible during
implementation; this is not a claim of live deployment validation.

Before production use, opt in manually to a diagnostic job, then ASE/EMT, then
a production calculator. Verify the compute hostname, Python/library/device
configuration, job ID, actual artifacts, and monitoring after restart. Record
the transcript and run evidence. CUDA gRASPA is deferred until the user-supplied
integration is available; the existing SYCL/H2O implementation is unchanged.

References: [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/),
[IRI reference compute models](https://github.com/doe-iri/iri-facility-api-python/blob/main/app/routers/compute/models.py),
[IRI reference task models](https://github.com/doe-iri/iri-facility-api-python/blob/main/app/routers/task/models.py).
