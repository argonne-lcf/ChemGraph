# ASE workflows on Polaris from a login node

Run ChemGraph and a persistent ASE MCP server on a Polaris login node. The
server uses Parsl's `PBSProProvider` to acquire compute nodes and reuse their
GPU workers across calculations. Skills guide the requests; the existing
ChemGraph ASE engine performs the science. A direct PBS example is also included.

These examples have hermetic tests; a real Polaris run has **not yet been
validated**. Record the actual job IDs, transcript, and artifacts during site validation.

## Prepare the environment

Use a shared project directory accessible from login and compute nodes and a
ChemGraph installation containing these examples. Both the server and compute
environment need `chemgraph[parsl]`, CUDA-enabled PyTorch, MACE, and the optional
MACE-Polar add-on. For a matching checkout, install the add-on with:

```bash
python -m pip install -r requirements/mace-polar.txt
```

Follow the [Polaris Parsl guide](https://docs.alcf.anl.gov/polaris/workflows/parsl/)
for site modules and environment setup. Create an initialization script that
loads those modules and activates your shared environment without prompting.
Supply its absolute path. The login-side server uses the same environment so
its calculator schemas match the workers' capabilities.

Stage an explicit local MACE-Polar `polar-1-m` model file before submitting
calculations. Use the [MACE foundation-model instructions](https://github.com/ACEsuit/mace)
for the installed engine version. The calculation runner records the model's
path and SHA-256 and never downloads weights. Configure your LLM provider in
the login environment; the compute workers do not call an LLM.

## Start the persistent ASE MCP server

Create a shared workspace and copy the installed templates into it. This step
only copies small package resources; it does not initialize a calculator:

```bash
WORKDIR=/eagle/YOUR_PROJECT/YOUR_USER/chemgraph-demo
mkdir -p "$WORKDIR"
cd "$WORKDIR"
python - <<'PY'
from importlib.resources import files
from pathlib import Path
root = files("chemgraph.skills")
for source, target in {
    "pbs-hpc/assets/polaris-parsl.toml.template": "execution.toml",
    "chemgraph/assets/water.xyz": "water.xyz",
    "chemgraph/assets/water-ase.json.template": "input.template.json",
}.items():
    with Path(target).open("xb") as stream:
        stream.write(root.joinpath(source).read_bytes())
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
worker_init = "source '/absolute/path/to/environment.sh'"
```

Confirm current [queue limits](https://docs.alcf.anl.gov/polaris/running-jobs/)
and declare every filesystem used by your inputs, environment, and outputs.
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

## Ask ChemGraph for calculations

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

The same `run_ase_single` input supports these drivers:

| Driver | Behavior |
| --- | --- |
| `opt` | Optimize the geometry |
| `vib` | Optimize and calculate frequencies/normal modes |
| `ir` | Optimize and calculate frequencies plus an IR spectrum |
| `thermo` | Optimize, calculate vibrations, and compute ideal-gas thermochemistry |

The input template has explicit calculator/model settings and the driver's
parameters. `thermo` already includes optimization and vibrations; it does not
produce IR. The combined prompt therefore submits a `thermo` task and an `ir`
task. Each uses its own directory; the engine currently repeats their shared
preparatory work. MACE-Polar supports dipoles needed for IR; MACE-OFF does not.
The model and method choices remain explicit for other molecules.

## Monitor and collect results

MCP returns `status="submitted"` and a **calculation batch ID** without waiting
for PBS. Use `check_job_status` and `get_job_results` with that ID. Use
`get_execution_status` for **PBS allocation IDs** and Parsl's cached scheduler
states. Several calculations can share an allocation. For scheduler diagnosis,
use `qstat -f PBS_JOB_ID` and `qstat -xf PBS_JOB_ID`, inspecting comments and logs.

Each isolated calculation directory contains `ase_input.json`, `calculation.log`,
`run_summary.json`, the requested result JSON, `final.xyz`, and driver-specific
artifacts: optimization trajectories, frequency CSVs, normal-mode trajectories,
and IR plots/spectrum/peak CSVs. The summary records the driver, compute host,
PBS ID, model path/hash, potential energy in eV, convergence, optimization steps,
timestamps, and artifact paths. Full thermochemistry and spectrum metadata are
in the result JSON. Artifacts use absolute paths on the shared filesystem.

A completed batch means futures finished; check each result's status and
convergence. Nonconverged calculations retain artifacts and report
`status="not_converged"`. Missing files or scheduler records leave the outcome
unresolved. The process exits 0 for successful convergence, 2 for nonconvergence,
and 1 for failure; early precondition failures may leave only the calculation log.

Keep the MCP server alive while tasks are outstanding. Metadata and saved
results survive restart, but in-flight Parsl futures cannot be reattached by
the current tracker. Inspect saved files and PBS evidence before retrying.
Batch cancellation attempts to cancel pending tasks; it does not guarantee
termination of a running calculation or deletion of its shared allocation.
Stopping the server normally releases its Parsl workers and allocations.

## Direct PBS alternative

For one batch calculation, ask the Deep Agent to use the recipe's **direct PBS
alternative**, specifying the desired driver and the same deployment inputs.
It stages `run_ase.py`, `job.pbs`, `submit_ase.sh`, `water.xyz`, and `input.json`
in a fresh shared run directory. These are packaged under the `chemgraph` and
`pbs-hpc` skills. Fill every placeholder with correctly escaped JSON or shell
paths. The batch script sources the compute environment and runs:

```bash
python run_ase.py --input input.json --require-pbs
```

From the **login node**, validate and submit the completed batch script:

```bash
bash -n job.pbs
bash submit_ase.sh
qstat -f "$(cat job.id)"
```

The submission helper records its attempt before calling `qsub`, saves `job.id`
and `qsub.stderr`, and refuses a repeated attempt in that directory. Preserve the
marker when submission is uncertain and inspect PBS before a replacement job.
The compute helper requires PBS metadata and a hostname listed in `PBS_NODEFILE`.
It uses the same ASE drivers and result summaries as the MCP example. This path
does not use the MCP server or Parsl-managed allocation.
