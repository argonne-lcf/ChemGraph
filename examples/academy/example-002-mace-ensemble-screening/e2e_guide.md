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
python -m pip install -e ".[academy,parsl]"
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

Stage the MACE model:

```bash
cd "$LOCAL_CHEMGRAPH"

MODEL=src/chemgraph/academy/campaigns/example-002-mace-ensemble-screening/models/mace-mpa-0-medium.model
URL=https://github.com/ACEsuit/mace-mp/releases/download/mace_mpa_0/mace-mpa-0-medium.model

mkdir -p "$(dirname "$MODEL")"
test -f "$MODEL" || curl -L --fail -o "$MODEL" "$URL"
ls -lh "$MODEL"
```

Then sync ChemGraph again.

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

export CHEMGRAPH_EXECUTION_BACKEND=parsl
export COMPUTE_SYSTEM="$ALCF_SYSTEM"

export PATH="$REMOTE_ROOT/bin:$REMOTE_ROOT/tools/redis/bin:$PATH"

: "${CHEMGRAPH_EXECUTION_BACKEND:?must be set to 'parsl' before launch}"
: "${COMPUTE_SYSTEM:?must be set to aurora or polaris before launch}"
echo "execution backend = $CHEMGRAPH_EXECUTION_BACKEND"
echo "compute system    = $COMPUTE_SYSTEM"

chemgraph academy run-compute \
  --system "$ALCF_SYSTEM" \
  --run-id "$RUN_ID" \
  --campaign mace-ensemble-screening-20 \
  --lm-user "$ARGO_USER"
```

If you reconnect to the login/compute node and re-run only the final
`chemgraph academy run-compute` invocation, the env exports above will not be
in your shell. Re-run the full block, or re-export both variables, before
relaunching. If `CHEMGRAPH_EXECUTION_BACKEND` is unset, the MCP server can fall
back to LocalBackend and produce `BrokenProcessPool` failures under per-rank
memory pressure.

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

If MACE results come back as `PicklingError: Can't pickle
run_mace_singleArguments`, the remote ChemGraph checkout does not have the
worker-module fix synced. Sync the latest ChemGraph checkout to the ALCF
filesystem, restart the dashboard with a fresh run id, and rerun from a fresh
compute allocation.

If MACE results come back as `BrokenProcessPool` failures, confirm the MACE MCP
server initialized Parsl:

```bash
grep "backend initialised" \
  "$REMOTE_ROOT/runs/$RUN_ID/rank3/mcp_logs/mace.log"
```

Expected:

```text
CGFastMCP backend initialised: ParslBackend
```

If the log shows `LocalBackend initialized with 4 workers`, re-run the full
compute block with `CHEMGRAPH_EXECUTION_BACKEND=parsl`.

If the log shows `Parsl is required for the ParslBackend`, the Parsl package is
missing from the venv:

```bash
python -m pip install -e ".[academy,parsl]"
```
