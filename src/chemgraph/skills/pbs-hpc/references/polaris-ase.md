# ASE calculations on Polaris

Read [Polaris](polaris.md) for site setup. ChemGraph and its LLM calls stay on
the login node; ASE calculations execute on PBS compute nodes. Prefer direct
PBS for individual jobs when `execute` runs on the submission host. Use the
optional MCP/Parsl route for worker reuse or an explicit user request.

## Choose the calculation

| Driver | Calculation |
| --- | --- |
| `opt` | Geometry optimization |
| `vib` | Optimization followed by vibrational frequencies and normal modes |
| `ir` | Optimization, frequencies, and an IR spectrum; requires calculator dipoles |
| `thermo` | Optimization, vibrations, and ideal-gas thermochemistry at the supplied temperature/pressure |

An optimization-plus-frequencies request needs one `vib` job. Use the requested
driver rather than submitting redundant prerequisites.
`thermo` does not also produce an IR spectrum. Keep each requested calculation
in its own output directory. A separate `ir` task is needed when both IR and
thermochemistry are requested; the current engine does not reuse their Hessians.

The example uses neutral singlet water and MACE-Polar (`mace_polar`, local
`polar-1-m` checkpoint, CUDA, float64, charge 0, multiplicity 1). The matching
`graph-longrange` add-on must be installed in the compute environment. Preserve
explicit calculator choices: MACE-OFF lacks the dipoles required by `ir`.
EMT is useful only for hermetic infrastructure tests, not the scientific example.

## Direct PBS workflow

Obtain the project account, queue/walltime, shared run directory, compute
environment initialization script, compute Python executable, and local model
path. The login environment needs ChemGraph and its configured LLM provider;
the compute environment needs ChemGraph and the requested calculator dependencies.
Direct submission uses the shell's PBS commands and does not require MCP or Parsl.

Launch the standalone Deep Agent on the login node with a real shared workspace:

```bash
chemgraph run --interactive --workflow deep_agent \
  --deepagent-workspace "$WORKDIR" --model "$LLM_MODEL"
```

Copy the [ASE entrypoint](../../chemgraph/scripts/run_ase.py),
[PBS template](../assets/polaris-ase.pbs.template), and
[submission helper](../scripts/submit_ase.sh) into a fresh shared run directory
as `run_ase.py`, `job.pbs`, and `submit_ase.sh`. Stage the requested structure
(or [example water](../../chemgraph/assets/water.xyz)) and fill
[the input template](../../chemgraph/assets/water-ase.json.template) as `input.json`.
Use `ASEInputSchema` fields for the requested calculator and scientific settings,
with absolute input, model, and result paths. Place the result JSON in this run
directory. Serialize JSON; shell-quote environment and Python paths in `job.pbs`.
Fill account, job name, and filesystems. The example requests one node for 30
minutes in `debug`, with one process using one GPU; adapt queue/walltime to the
request and current site limits. Keep `--require-pbs` on the runner command.

Preserve the template's proxy and `TMPDIR=/tmp` exports after sourcing the
compute environment. Stage the model beforehand; never execute the calculation
helper on the login node to initialize a calculator or download a model.

Inspect the completed files, check JSON/schema values, and run `bash -n job.pbs`.
From the real host run directory, submit with `bash submit_ase.sh`, following
existing action approvals. The helper preserves `submission.started`, `job.id`,
and `qsub.stderr` and refuses a repeated attempt. Errors/empty IDs can indicate
an uncertain submission: inspect PBS records, matching job name and directory;
do not delete the marker or automatically submit a replacement.

### Monitor or resume inspection

Read `job.id` and inspect that same job with `qstat -f JOB_ID`; use
`qstat -xf JOB_ID` for retained history after it leaves the active queue.
Report queued, held, or running states with the ID and run directory between
checks. Inspect scheduler comments, PBS stdout/stderr, `run_summary.json`, and
the configured result JSON before reporting success. Missing scheduler history
or results leaves the outcome unresolved. Cancel only the requested job with
`qdel JOB_ID` when cancellation is requested.

A later agent session needs only the shared run directory: read its existing
`job.id`, input, and result files and inspect PBS. No live agent session or MCP
server is needed to keep an accepted direct batch job running. Do not submit
again to recover monitoring. Preserve partial artifacts and the submission marker
on failure; an explicitly requested retry uses a fresh directory after resolving
the prior job's state.

## Optional: persistent MCP server and PBS-managed Parsl

Obtain the project account, shared workspace, compute environment initialization
script, and local model path. Stage [the configuration](../assets/polaris-parsl.toml.template)
as `execution.toml`, [water](../../chemgraph/assets/water.xyz) as `water.xyz`, and
[the calculation template](../../chemgraph/assets/water-ase.json.template) as
the input reference. Fill paths with JSON/TOML serialization; shell-quote the
environment script path in `worker_init`. The environment must include
ChemGraph, Parsl, CUDA PyTorch, MACE, and the MACE-Polar add-on on both the server
and workers so calculator schemas match.

Preserve the template's compute-node proxy and temporary-directory exports
after sourcing the environment:

```bash
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export TMPDIR=/tmp
```

With `PBSProProvider`, place these in `worker_init`. With `LocalProvider`, set
them in the PBS job script before the Python driver starts. `TMPDIR=/tmp`
avoids the single-node `OSError: AF_UNIX path too long` documented in
[ALCF's Parsl guide](https://docs.alcf.anl.gov/polaris/workflows/parsl/#known-issues).
The direct PBS template also exports these values before launching Python.

The user starts the ASE MCP server in a persistent login-node session with
`--pbs-workers --execution-config /absolute/path/execution.toml` and attaches
ChemGraph via its streamable-HTTP URL. Keep server and client on the same login
host for a loopback URL. Use a separate `--jobs-file` for this server.

1. Read attached schemas; call `run_ase_single` with an `ASEInputSchema` payload
   and an absolute `output_results_file` in a fresh per-calculation directory.
   Ensembles automatically add per-structure directories in PBS-worker mode.
2. Save the returned `batch_id`. The tool returns immediately while PBS queues
   workers; `check_job_status` / `get_job_results` inspect that same batch.
3. Use `get_execution_status` for Parsl allocation IDs and cached states. A batch
   ID is not a PBS job ID; multiple calculations can share one allocation.
   Use `qstat -f` / `qstat -xf` with a returned **scheduler** ID when diagnosing PBS.
4. Read calculation summaries and artifacts. Return pending state between
   checks instead of resubmitting. Stop only the requested pending calculation;
   cancelling a batch is not equivalent to deleting the entire allocation.

The default pool has one node and one GPU worker, at most one concurrent PBS
allocation, and releases idle capacity after approximately 120 seconds. It can
acquire replacement capacity for later tasks; `max_blocks=1` is a concurrency
limit, not a lifetime submission limit. Increase workers to at most four for
independent calculations. Each frequency/IR calculation remains one task;
finite-difference displacements are not distributed by this configuration.

Parsl owns allocation submission here; do not also invoke the direct `qsub`
helper. Keep the MCP server alive for in-flight futures. Persisted metadata and
result files survive, but outstanding Parsl futures cannot be reattached after
server restart. Inspect saved artifacts and scheduler evidence before any retry.
Normal server shutdown releases its workers/allocations.

## Results and failures

Each process uses its output directory for logs, modes, and spectra.
`run_summary.json` includes the driver, PBS ID, compute hostname, calculator/model
path and SHA-256, potential energy in eV, convergence, optimization steps, and
artifact paths. The full configured result JSON includes frequencies, IR data,
or thermochemistry as appropriate; `final.xyz` and trajectories retain geometry.

The runner exits 0 only for a successful, converged calculation, 2 for
nonconvergence, and 1 for failure. It preserves partial results. Preconditions
rejected before calculation setup may produce only stderr or `calculation.log`.
An MCP batch completing means futures finished, not that every calculation
succeeded: inspect individual results and convergence. Missing scheduler history
or result files means unresolved evidence, not completion.
