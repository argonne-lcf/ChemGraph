# Example 002 E2E Guide

This guide runs the `mace-ensemble-screening-20` ChemGraph Academy campaign on
Aurora or Polaris. The campaign starts five persistent logical agents under MPI:

```text
coordinator-agent
structure-agent-a
structure-agent-b
mace-agent
assessment-agent
```

The coordinator delegates 20 SMILES candidates, structure agents generate XYZ
files, the MACE agent runs an ensemble energy screen, and the assessment agent
summarizes readiness/ranking evidence.

## About The MACE Path

This example deliberately runs MACE through the general `run_ase` tool
(`chemgraph.mcp.mcp_tools`), which executes MACE in-process inside the MCP
server. It does **not** exercise `chemgraph.mcp.mace_mcp_hpc` or the
Parsl/EnsembleLauncher/Globus Compute backends — those are being reworked in
a separate PR. Once that lands and the WorkerLost subprocess fix is folded
back in, this example can be switched back to the HPC MACE path.

In-process MACE means each per-structure energy evaluation runs synchronously
in the mace-agent's MCP server process. A 20-structure screen completes in
a few minutes on CPU.

## Configure Paths

Set these values in each terminal before copying the commands below:

```bash
export ALCF_PROJECT=<project-name>
export ALCF_USER=<shared-filesystem-user>
export ALCF_LOGIN=<ssh-login>
export ARGO_USER=<argo-user>

export LOCAL_CHEMGRAPH=<local-chemgraph-checkout>
```

For Aurora:

```bash
export ALCF_SYSTEM=aurora
export ALCF_HOST=aurora.alcf.anl.gov
export REMOTE_ROOT=/flare/$ALCF_PROJECT/$ALCF_USER
```

For Polaris:

```bash
export ALCF_SYSTEM=polaris
export ALCF_HOST=polaris.alcf.anl.gov
export REMOTE_ROOT=/eagle/$ALCF_PROJECT/$ALCF_USER
```

`ALCF_USER` is the shared-filesystem path component. It may differ from the SSH
login and from the Argo user.

## One-Time Setup

Sync ChemGraph:

```bash
cd "$LOCAL_CHEMGRAPH"

rsync -az --delete --delete-excluded \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'runs/' \
  --exclude 'venvs/' \
  --exclude '*.pyc' \
  ./ \
  "$ALCF_LOGIN@$ALCF_HOST:$REMOTE_ROOT/ChemGraph/"
```

Install ChemGraph dependencies on the remote system:

```bash
ssh "$ALCF_LOGIN@$ALCF_HOST"
cd "$REMOTE_ROOT/ChemGraph"

# Aurora:
module load frameworks

# Polaris:
# module use /soft/modulefiles
# module load conda
# conda activate base

source "$REMOTE_ROOT/venvs/academy-swarm/bin/activate"
python -m pip install -e ".[academy]"
```

Verify the campaign is visible:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python -m chemgraph.cli.main academy campaigns
```

Expected:

```text
mace-ensemble-screening-20
```

Verify Redis:

```bash
export PATH="$REMOTE_ROOT/tools/redis/bin:$PATH"
command -v redis-server
redis-server --version
```

If Redis is missing, build it once on a login/UAN node:

```bash
cd "$REMOTE_ROOT"
mkdir -p src tools
cd src
test -d redis || git clone --depth 1 https://github.com/redis/redis.git
cd redis
make -j4
make PREFIX="$REMOTE_ROOT/tools/redis" install
```

The `mace_mp` calculator downloads its foundation model on first use into
`~/.cache/mace`, so no manual MACE-model staging is needed for this example.
First-call download can take a minute; pre-warm it once on the compute node
to skip that wait at run time. The compute node only reaches external sites
through the ALCF outbound proxy, so set the proxy env vars first:

```bash
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
python -c "from mace.calculators import mace_mp; mace_mp(model='medium-mpa-0', device='cpu')"
```

## Start argo-shim

On the local machine:

```bash
CELS_USERNAME="$ARGO_USER" \
PYTHONPATH=<argo-shim-checkout> \
python -m argo_shim --no-auth --no-update-settings --port 18085
```

## Start Dashboard

Use a fresh run id:

```bash
cd "$LOCAL_CHEMGRAPH"

export RUN_ID="${ALCF_SYSTEM}-mace-ensemble-screening-001"

PYTHONPATH=src python -m chemgraph.cli.main academy dashboard -- \
  --system "$ALCF_SYSTEM" \
  --remote-host "$ALCF_LOGIN@$ALCF_HOST" \
  --campaign mace-ensemble-screening-20 \
  --lm-connect mac-argo-relay \
  "$RUN_ID"
```

The dashboard command starts the local dashboard, an rsync mirror, an SSH
control connection, and a relay from compute nodes to local `argo-shim`.

## Start The Campaign On Compute

Run inside an interactive allocation:

```bash
cd "$REMOTE_ROOT/ChemGraph"

# Aurora:
module load frameworks

# Polaris:
# module use /soft/modulefiles
# module load conda
# conda activate base

source "$REMOTE_ROOT/venvs/academy-swarm/bin/activate"

export RUN_ID="${ALCF_SYSTEM}-mace-ensemble-screening-001"

export NUMEXPR_MAX_THREADS=256
export NUMEXPR_NUM_THREADS=64
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Aurora/Polaris compute nodes reach external sites (GitHub, S3) only
# through the ALCF outbound proxy. Without these, mace_mp(model="medium-mpa-0")
# hangs trying to fetch the foundation model on first use.
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1"

export PATH="$REMOTE_ROOT/bin:$REMOTE_ROOT/tools/redis/bin:$PATH"

chemgraph academy run-compute \
  --system "$ALCF_SYSTEM" \
  --run-id "$RUN_ID" \
  --campaign mace-ensemble-screening-20 \
  --lm-user "$ARGO_USER"
```

If the wrapper is installed but `chemgraph` is not on `PATH`, use:

```bash
chemgraph-academy-run \
  --system "$ALCF_SYSTEM" \
  --run-id "$RUN_ID" \
  --campaign mace-ensemble-screening-20 \
  --lm-user "$ARGO_USER"
```

## Reopen A Local Dashboard

Once the run has been synced locally:

```bash
cd "$LOCAL_CHEMGRAPH"

PYTHONPATH=src python -m chemgraph.cli.main academy dashboard -- \
  --system "$ALCF_SYSTEM" \
  --remote-host "$ALCF_LOGIN@$ALCF_HOST" \
  --campaign mace-ensemble-screening-20 \
  "$RUN_ID" \
  --local
```

## Dashboard For Traditional ChemGraph Runs

The dashboard also renders single-agent ChemGraph runs that were not launched
through Academy. Pass `--trace-dir <path>` to `chemgraph run` to write the
events the dashboard needs (`events.jsonl`, `status.json`, `manifest.json`),
then point the dashboard at that directory.

On-site at ANL, the simplest path is the built-in Argo support — no shim or
relay needed (set `ARGO_USER` once per shell, or in your shell profile):

```bash
export ARGO_USER="$ARGO_USER"

chemgraph run \
  -q "What is the SMILES for water" \
  -m "argo:gpt-5.4" \
  --trace-dir ./run-001
```

Then serve the trace directory:

```bash
chemgraph dashboard -- --run-dir ./run-001 --port 8765
# Open http://127.0.0.1:8765
```

The browser shows the same per-agent workflow inspector that Academy displays
for a logical-agent node (query → LLM call → tool calls → output), but at the
top level since the run only has one agent. Use a fresh `--trace-dir` per run
so multiple runs don't pile into one `events.jsonl`.

`--trace-dir` is currently only effective for the `single_agent` workflow.
Other workflows (`multi_agent`, `python_relp`, `graspa`, `rag_agent`,
`single_agent_xanes`, ...) run normally but don't yet emit dashboard events,
and the CLI prints a yellow warning for those.

If the browser shows "Waiting for ChemGraph workflow execution events" after a
run completed successfully, the remote checkout is missing the
`llm_decision`-on-every-LLM-call fix. Sync the latest ChemGraph and clear
stale bytecode locally:

```bash
find src/chemgraph -name __pycache__ -type d -exec rm -rf {} +
```

## Troubleshooting

Check the relay from compute:

```bash
UAN_RELAY_HOST="$(tr -d '[:space:]' < "$REMOTE_ROOT/uan-relay-18186.host")"
curl --noproxy '*' -I "http://${UAN_RELAY_HOST}:18186/v1/models"
```

Expected:

```text
HTTP/1.1 200 OK
```

If the first model response is an Argo access-denied notice for `<argo-user>`,
the compute command was launched without `--lm-user "$ARGO_USER"`. Use a fresh
run id, or restart the dashboard with `--overwrite-run`, then rerun compute
with `--lm-user`.

If imports are slow or NumExpr complains, set:

```bash
export NUMEXPR_MAX_THREADS=256
export NUMEXPR_NUM_THREADS=64
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

If MACE energy evaluations are slow, the first call per worker pays a
one-time foundation-model download into `~/.cache/mace`. Pre-warm by
running the snippet under "About The MACE Path" above on the compute node
before launching the campaign.
