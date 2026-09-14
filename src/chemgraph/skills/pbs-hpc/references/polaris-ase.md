# ASE calculations on Polaris

Read [Polaris](polaris.md) for site setup. ChemGraph and its LLM calls stay on
the login node; ASE calculations execute on PBS compute nodes.

## Choose the calculation

| Driver | Calculation |
| --- | --- |
| `opt` | Geometry optimization |
| `vib` | Optimization followed by vibrational frequencies and normal modes |
| `ir` | Optimization, frequencies, and an IR spectrum; requires calculator dipoles |
| `thermo` | Optimization, vibrations, and ideal-gas thermochemistry at the supplied temperature/pressure |

Use the requested driver rather than submitting redundant prerequisites.
`thermo` does not also produce an IR spectrum. Keep each requested calculation
in its own output directory. A separate `ir` task is needed when both IR and
thermochemistry are requested; the current engine does not reuse their Hessians.

The example uses neutral singlet water and MACE-Polar (`mace_polar`, local
`polar-1-m` checkpoint, CUDA, float64, charge 0, multiplicity 1). The matching
`graph-longrange` add-on must be installed on the server and workers. Preserve
explicit calculator choices: MACE-OFF lacks the dipoles required by `ir`.
EMT is useful only for hermetic infrastructure tests, not the scientific example.

## Primary: persistent MCP server and PBS-managed Parsl

Obtain the project account, shared workspace, compute environment initialization
script, and local model path. Stage [the configuration](../assets/polaris-parsl.toml.template)
as `execution.toml`, [water](../../chemgraph/assets/water.xyz) as `water.xyz`, and
[the calculation template](../../chemgraph/assets/water-ase.json.template) as
the input reference. Fill paths with JSON/TOML serialization; shell-quote the
environment script path in `worker_init`. The environment must include
ChemGraph, Parsl, CUDA PyTorch, MACE, and the MACE-Polar add-on.

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

## Direct batch alternative

For a single batch job, copy the [ASE entrypoint](../../chemgraph/scripts/run_ase.py),
[PBS template](../assets/polaris-ase.pbs.template), and
[submission helper](../scripts/submit_ase.sh) into a fresh shared run directory
as `run_ase.py`, `job.pbs`, and `submit_ase.sh`, with `water.xyz` and `input.json`.
Fill the input template's driver and absolute paths, then PBS account, job name,
filesystems, and shell-quoted environment/Python paths. No LLM or MCP server
runs inside this batch job. Stage the model beforehand; never run this
calculation helper on the login node to download or initialize a model.

Inspect the completed files and run `bash -n job.pbs`. From the real host run
directory, submit with `bash submit_ase.sh`, following existing action approvals.
The helper preserves `submission.started`, `job.id`, and `qsub.stderr` and refuses
a repeated attempt. Errors/empty IDs can indicate an uncertain submission:
inspect PBS records, matching job name and directory; do not delete the marker
or automatically submit a replacement. A missing job is not proof of success.

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
