# ASE workflows on Polaris from a login node

Run ChemGraph's Deep Agent on a Polaris login node and use its bundled skills
to prepare, submit, and monitor an ASE calculation through PBS. This is the
recommended route for individual geometry optimizations and frequency jobs.
The agent's shell calls the packaged submission helper; the existing ChemGraph
ASE engine performs the science on a compute node. No ASE MCP server or Parsl
is required. An optional MCP/Parsl route below supports reusable worker pools.

These examples have hermetic tests; a real Polaris run has **not yet been
validated**. Record the actual job IDs, transcript, and artifacts during site validation.

## Prepare the environment

Use a shared project directory accessible from login and compute nodes and a
ChemGraph installation containing the ASE runner and packaged skills. The login
environment needs ChemGraph and its configured LLM provider, with `qsub` and
`qstat` available to its shell.
The compute environment needs ChemGraph and the requested calculator dependencies.
For this MACE-Polar water example, use CUDA-enabled PyTorch, MACE, and the optional
MACE-Polar add-on. In a matching checkout, install the add-on with:

```bash
python -m pip install -r requirements/mace-polar.txt
```

Follow the [Polaris Parsl guide](https://docs.alcf.anl.gov/polaris/workflows/parsl/)
for site modules and environment setup. Create an initialization script that
loads those modules and activates your shared environment without prompting.
Supply its absolute path and the absolute Python executable for that environment.
Only the optional MCP/Parsl route needs `chemgraph[parsl]`; its server and workers
also need matching calculator dependencies so their schemas agree.

Stage an explicit local MACE-Polar `polar-1-m` model file before submitting
calculations. Use the [MACE foundation-model instructions](https://github.com/ACEsuit/mace)
for the installed engine version. The calculation runner records the model's
path and SHA-256 and never downloads weights. Configure your LLM provider in
the login environment; the compute workers do not call an LLM.

## Choose the calculation

| Driver | Behavior |
| --- | --- |
| `opt` | Optimize the geometry |
| `vib` | Optimize and calculate frequencies/normal modes |
| `ir` | Optimize and calculate frequencies plus an IR spectrum |
| `thermo` | Optimize, calculate vibrations, and compute ideal-gas thermochemistry |

For optimization and frequencies together, submit one `vib` job. `thermo`
already includes both steps but does not produce IR; requesting thermochemistry
and IR needs separate `thermo` and `ir` calculations in separate directories.
The engine currently repeats their shared preparatory work. MACE-Polar supports
dipoles needed for IR; MACE-OFF does not. Preserve the user's calculator and
scientific settings for other molecules. EMT is only an infrastructure test
calculator for this walkthrough, not a substitute for the requested model.

## Direct PBS jobs

Create a fresh shared workspace. The agent can stage the bundled resources
from its skills; the following commands also let you stage them manually.
They only copy small package resources and do not initialize a calculator:

```bash
WORKDIR=/eagle/YOUR_PROJECT/YOUR_USER/chemgraph-demo
mkdir -p "$WORKDIR"
cd "$WORKDIR"
python - <<'PY'
from importlib.resources import files
from pathlib import Path
root = files("chemgraph.skills")
for source, target in {
    "chemgraph/scripts/run_ase.py": "run_ase.py",
    "pbs-hpc/assets/polaris-ase.pbs.template": "job.pbs",
    "pbs-hpc/scripts/submit_ase.sh": "submit_ase.sh",
    "chemgraph/assets/water.xyz": "water.xyz",
    "chemgraph/assets/water-ase.json.template": "input.json",
}.items():
    with Path(target).open("xb") as stream:
        stream.write(root.joinpath(source).read_bytes())
PY
```

Use a fresh directory for each calculation. Complete `input.json` with the
driver, absolute structure/model/result paths, and scientific settings using
`ASEInputSchema`. Place the result JSON in this run directory. Complete `job.pbs`
with the project account, job name, filesystems, and shell-quoted environment
and Python paths. Use JSON serialization for input values and `shlex.quote`
for shell paths. The agent can fill these files from the prompt below.

The template requests one node for 30 minutes in `debug`, with one process
using one GPU. Confirm current [queue limits](https://docs.alcf.anl.gov/polaris/running-jobs/)
and adapt the queue/walltime to the calculation. Declare every filesystem used
by inputs, environment, and outputs. Keep the proxy exports and `TMPDIR=/tmp`
after environment activation. Keep `--require-pbs`: the runner checks PBS
metadata and verifies that its hostname appears in `PBS_NODEFILE` before setup.

### Ask ChemGraph to prepare and submit

On the **login node**, with your configured LLM provider:

```bash
chemgraph run --interactive --workflow deep_agent \
  --deepagent-workspace "$WORKDIR" --model "$LLM_MODEL"
```

Bundled skills load automatically. The CLI enables a host shell with its existing
action approvals. Replace the deployment placeholders in this example prompt:

> Read the chemgraph and pbs-hpc skills and their Polaris ASE recipe. Use direct
> PBS to optimize water.xyz and calculate its vibrational frequencies in one
> vib job. Use MACE-Polar with model /absolute/path/to/polar-1-m.model, CUDA,
> float64, charge 0, multiplicity 1, BFGS, fmax 0.01 eV/Å, and 200 steps.
> Use project YOUR_PROJECT, queue debug, walltime 00:30:00, and filesystems
> home:eagle. The compute initialization script is /absolute/path/to/environment.sh
> and its Python is /absolute/path/to/environment/bin/python. Stage any missing
> bundled resources and complete input.json and job.pbs in this fresh workspace,
> with result.json here.
> Validate the files and submit once using submit_ase.sh. Save the PBS job ID
> and report its state and run directory. Read the actual results when available.

For geometry optimization alone, ask for `opt`. The batch script runs
`python run_ase.py --input input.json --require-pbs` on a compute node.
Do not execute this calculation command on the login node.

After completing and inspecting the files, the submission commands from the
real host run directory are:

```bash
bash -n job.pbs
bash submit_ase.sh
```

The helper records the attempt in `submission.started` before calling `qsub`,
saves `job.id` and `qsub.stderr`, and refuses a repeated attempt. A failed call
or missing ID can mean an uncertain submission; preserve the marker and inspect
PBS records using the job name and directory before considering a replacement.

### Monitor now or from another agent session

Read the saved full PBS job ID and inspect that same job:

```bash
qstat -f "$(cat job.id)"
```

After the job leaves the active queue, inspect retained history with
`qstat -xf "$(cat job.id)"`. Report queued, held, or running states with the ID
and run directory. Inspect scheduler comments, PBS stdout/stderr, and the result
files before reporting completion. A missing job or missing results leaves the
outcome unresolved. Use `qdel JOB_ID` only for a requested cancellation.

An accepted direct PBS job runs independently of the agent process. To inspect
it later, launch the same CLI command with the existing workspace and ask:

> Inspect the existing PBS calculation in this workspace. Read job.id and
> input.json, check that job's status or history, and inspect run_summary.json
> and result.json when available. Report convergence, energy, frequencies,
> and artifact paths. Do not submit another job.

Inspection uses the saved files and scheduler evidence across agent sessions.
On failure, preserve the original artifacts and submission marker. An explicitly
requested retry uses a fresh directory after the prior job's state is resolved.

## Optional MCP/Parsl worker reuse

Use this route for ensembles or repeated calculations sharing allocated workers,
or when explicitly requested. Keep the direct workflow above for individual
jobs. In the shared workspace, copy the installed Parsl template:

```bash
python - <<'PY'
from importlib.resources import files
from pathlib import Path
template = files("chemgraph.skills").joinpath("pbs-hpc/assets/polaris-parsl.toml.template")
with Path("execution.toml").open("xb") as stream:
    stream.write(template.read_bytes())
PY
```

Complete `execution.toml`: project account, absolute Parsl run directory, and
shell-quoted environment initialization path. Its important settings are:

```toml
[execution]
backend = "parsl"
system = "polaris"

[execution.parsl]
allocation_mode = "pbs"
run_dir = "/eagle/YOUR_PROJECT/YOUR_USER/chemgraph-demo/parsl"
account = "YOUR_PROJECT"
queue = "debug"
walltime = "00:30:00"
filesystems = "home:eagle"
nodes_per_block = 1
max_blocks = 1
max_workers_per_node = 1
worker_init = '''
source '/absolute/path/to/environment.sh'
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export TMPDIR=/tmp
'''
```

The proxy variables enable outbound internet access on compute nodes; the
calculation still uses the staged local model. Keep `TMPDIR=/tmp` after
environment activation to avoid Parsl's `OSError: AF_UNIX path too long` on
single-node jobs. ALCF documents the [proxy settings](https://docs.alcf.anl.gov/polaris/running-jobs/#compute-node-access-to-the-internet)
and [Parsl workaround](https://docs.alcf.anl.gov/polaris/workflows/parsl/#known-issues).
For `PBSProProvider`, these exports belong in `worker_init`. When using
`LocalProvider` inside an existing allocation, put them in the PBS job script
before starting the Python driver; worker initialization alone is too late for
the driver's temporary paths. The bundled direct PBS template includes them.

The worker connection defaults to the login host's `bond0` address; set `address`
only when your deployment needs a different compute-reachable login address.
Backend/system environment variables override TOML; remove stale
`CHEMGRAPH_EXECUTION_BACKEND` or `COMPUTE_SYSTEM` overrides before starting.

In a persistent terminal session on the login node, start:

```bash
python -m chemgraph.mcp.ase_mcp_hpc --transport streamable_http \
  --host 127.0.0.1 --port 9005 --pbs-workers \
  --execution-config "$WORKDIR/execution.toml" \
  --jobs-file "$WORKDIR/ase-jobs.json"
```

The server can start before an allocation exists. The first calculation causes
Parsl to request capacity. Default configuration bounds the pool to one node
with one GPU worker; set `max_workers_per_node=4` to use all four GPUs for
independent calculations. Idle capacity is released after approximately 120
seconds. `max_blocks` limits concurrent allocations, not lifetime submissions.
Each frequency calculation remains one task; its finite-difference displacements
are not individually distributed by this configuration.

### Ask ChemGraph through MCP

In another terminal on the **same login host**, using the configured LLM provider:

```bash
chemgraph run --interactive --workflow deep_agent \
  --deepagent-workspace "$WORKDIR" --model "$LLM_MODEL" \
  --mcp-url http://127.0.0.1:9005/mcp/
```

Bundled skills load automatically. Replace the model path in this prompt:

> Read the chemgraph and pbs-hpc skills and their Polaris ASE recipe. Use the
> attached ASE MCP tools to calculate water's thermochemistry at 298.15 K and
> 101325 Pa and its IR spectrum. Use the workspace water.xyz with MACE-Polar,
> model /absolute/path/to/polar-1-m.model, CUDA, float64, charge 0, and multiplicity 1.
> Use BFGS, fmax 0.01 eV/Å, and 200 steps. Give each calculation a fresh output
> directory inside this workspace. Save the returned batch IDs and inspect
> those batches without resubmitting. Report convergence, energy, frequencies,
> thermochemistry, and links to the IR and geometry artifacts as they become available.

### Monitor MCP batches

MCP returns `status="submitted"` and a **calculation batch ID** without waiting
for PBS. Use `check_job_status` and `get_job_results` with that ID. Use
`get_execution_status` for **PBS allocation IDs** and Parsl's cached scheduler
states. Several calculations can share an allocation. For scheduler diagnosis,
use `qstat -f PBS_JOB_ID` and `qstat -xf PBS_JOB_ID`, inspecting comments and logs.

A completed batch means futures finished; check each result's status and
convergence. Parsl owns allocation submission here; do not also invoke the direct
submission helper for the same calculation.

Keep the MCP server alive while tasks are outstanding. Metadata and saved
results survive restart, but in-flight Parsl futures cannot be reattached by
the current tracker. Inspect saved files and PBS evidence before retrying.
Batch cancellation attempts to cancel pending tasks; it does not guarantee
termination of a running calculation or deletion of its shared allocation.
Stopping the server normally releases its Parsl workers and allocations.

## Results and failures for both routes

Each calculation produces `run_summary.json`, the configured result JSON,
`final.xyz`, and driver-specific artifacts: optimization trajectories, frequency
CSVs, normal-mode trajectories, and IR plots/spectrum/peak CSVs. The summary records
the driver, compute host, PBS ID, model path/hash, potential energy in eV,
convergence, optimization steps, timestamps, and artifact paths. Full frequency,
thermochemistry, and spectrum data are in the result JSON.

Direct jobs retain the staged `input.json` and PBS stdout/stderr. MCP jobs also
write `ase_input.json` and `calculation.log`. Use absolute result paths on the
shared filesystem and a fresh output directory for each calculation.

Nonconverged calculations retain artifacts and report `status="not_converged"`.
The runner exits 0 for successful convergence, 2 for nonconvergence, and 1 for
failure. Early precondition failures may leave only stderr or the calculation
log. Missing files or scheduler records leave the outcome unresolved; scheduler
completion alone does not establish scientific success.
