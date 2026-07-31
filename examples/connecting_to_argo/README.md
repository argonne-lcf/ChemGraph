# Connecting ChemGraph to Argo from an ALCF compute node

ALCF compute nodes cannot reach the public Argo endpoint
(`apps.inside.anl.gov`) directly. This example walks through two
topologies for making ChemGraph on a compute node talk to Argo via
[`argo-shim`](https://github.com/anl-shieldsdigitalservices/argo-shim):

1. **Login-node relay** (recommended for compute-node workflows) —
   `argo-shim --tunnel` runs on the login node, opens an SSH tunnel to
   CELS, and every compute node runs its own
   `argo-shim --tunnel-host <login>` to expose an HTTP endpoint locally.
2. **Laptop relay** — `argo-shim --relay <login>` runs on your laptop
   and creates a reverse tunnel via SSH to the login node. Compute
   nodes use their own `argo-shim --tunnel-host <login>` the same way.

Both patterns end with the same compute-node command: ChemGraph points
at `http://127.0.0.1:18085/argoapi/v1` and Argo responds through the
chain of hops.

## Topology (both patterns)

```
                     Argo endpoint (apps.inside.anl.gov)
                                 ^
                                 |  HTTPS (via CELS)
                                 |
        +------------------------+---------------------------+
        |                                                    |
        |  Pattern 1 (login-node relay):                     |
        |      argo-shim --tunnel   on login node            |
        |                                                    |
        |  Pattern 2 (laptop relay):                         |
        |      argo-shim --relay <login>  on your laptop     |
        |      (opens SSH reverse tunnel to login node)      |
        +----------------------------------------------------+
                                 ^
                                 |  TCP tunnel :18084 on login node
                                 |
                       compute node argo-shim
                       (--tunnel-host <login> --tunnel-port 18084)
                                 ^
                                 |  HTTP :18085 on compute-node loopback
                                 |
                              ChemGraph
```

The **login-node relay** is simpler if you can `ssh` from the login
node to CELS (with a key or Duo). The **laptop relay** is what you
want if only your laptop has the credentials CELS wants.

---

## Prereqs

- ALCF account with SSH access to Aurora (or another cluster with the
  same shim pattern). This guide uses Aurora; substitute
  `polaris.alcf.anl.gov` / `crux.alcf.anl.gov` if you're on those.
- CELS account (`homes.cels.anl.gov`) reachable via SSH — either
  password + Duo, or an SSH key registered on your CELS portal.
- `argo-shim` installed on the machine that runs `--tunnel` or
  `--relay`. On Aurora that's a small pip install into a venv:
  ```bash
  module load frameworks
  python3 -m venv ~/venvs/argo-shim
  source ~/venvs/argo-shim/bin/activate
  pip install argo-shim  # or `pip install -e /path/to/argo-shim` from source
  ```

## Pattern 1: Login-node relay

### Step 1a — start the tunnel on an Aurora login node

```bash
# On an Aurora UAN (aurora-uan-XXXX):
module load frameworks
source ~/venvs/argo-shim/bin/activate

# CELS_USERNAME is required when your Aurora login name differs from
# your CELS login name (e.g. "jinchuli" vs "jinchu.li"). If they match,
# you can omit this export.
export CELS_USERNAME=<your.cels.login>

argo-shim --no-auth --no-update-settings --port 18085 --tunnel
```

One Duo push. On success the shim prints:

```
Tunnel created on port 18084 (bound to 0.0.0.0)
On the compute node, run:
  argo-shim --tunnel-host <UAN> --port 18085 --tunnel-port 18084
```

Note the login node's hostname (`hostname -f` or the printout above)
— compute nodes need to reach it explicitly. Aurora's `aurora.alcf.anl.gov`
load-balances SSH across multiple UANs so pin the specific one.

The command exits after creating the tunnel; the SSH connection stays
backgrounded (`ssh -N -f`). Verify with:

```bash
ss -tlnp | grep 18084   # want: LISTEN 0.0.0.0:18084 ... ssh
```

### Step 2a — start the compute-node shim

Inside a PBS interactive shell (`qsub -I ...`) or PBS script:

```bash
module load frameworks
source ~/venvs/argo-shim/bin/activate
export CELS_USERNAME=<your.cels.login>

# Pin the specific UAN where you started the tunnel in step 1a
export UAN_HOST=aurora-uan-XXXX.hostmgmt.cm.aurora.alcf.anl.gov

# CRITICAL: Aurora compute nodes route outbound HTTP through Squid via
# http_proxy=http://proxy.alcf.anl.gov:3128 by default. Without this
# no_proxy exemption, ChemGraph's request to the UAN hits Squid instead
# of the tunnel and Argo is unreachable.
export no_proxy="${UAN_HOST},127.0.0.1,localhost,.alcf.anl.gov,*.alcf.anl.gov"
export NO_PROXY="$no_proxy"

# Start the HTTP proxy layer that ChemGraph will POST to
nohup argo-shim --no-auth --no-update-settings \
  --port 18085 --tunnel-host $UAN_HOST --tunnel-port 18084 \
  > /tmp/argo-shim.log 2>&1 &

sleep 5
curl -s http://127.0.0.1:18085/v1/models | head -c 100
# Expect JSON with the model list.
```

### Step 3a — point ChemGraph at 127.0.0.1

See [`test_run.py`](test_run.py) for the runnable script. Minimum
form:

```python
from chemgraph.agent.llm_agent import ChemGraph

cg = ChemGraph(
    model_name="argo:gpt-4.1-mini",              # lowercase!
    workflow_type="single_agent",
    base_url="http://127.0.0.1:18085/argoapi/v1",
    api_key="dummy",
    argo_user="<your.cels.login>",
)
```

Two things this script gets right that trip most people up on first
try:

- **Lowercase `argo:gpt-4.1-mini`.** The `argo:`-prefixed name has to
  match `supported_argo_models` exactly (case-sensitive), which lists
  lowercase. Mixed case falls into a generic vLLM branch that doesn't
  wire `argo_user`, and Argo returns 500.
- **`argo_user` matches your CELS login.** Argo requires this in the
  request body. Missing → 500. Wrong value → 401.

Run the script (see below). Expected: LLM response, either a direct
answer or a tool call to `run_ase`.

---

## Pattern 2: Laptop relay

If the login node can't SSH to CELS but your laptop can, run
`argo-shim --relay` from your laptop. It creates the tunnel on your
laptop, then reverse-forwards the tunnel to the specified login node
so compute nodes still see it at `<login>:18084`.

### Step 1b — on your laptop

```bash
argo-shim --no-auth --no-update-settings --port 18085 \
  --relay <sshuser>@aurora.alcf.anl.gov
```

You'll need SSH access from laptop to the login node AND from laptop
to CELS. Duo prompt is on the CELS hop only. After success the
compute-node instructions are the same as **Step 2a** above — the
compute-side shim doesn't care whether the login-node tunnel was
opened by `--tunnel` (Pattern 1) or `--relay` (Pattern 2).

---

## Common failure modes

| Symptom | Fix |
|---|---|
| `ERR_CONNECT_FAIL` HTML response | `no_proxy` doesn't exempt the UAN hostname → Squid intercepts. Re-check `echo $no_proxy` on the compute node. |
| `openai.InternalServerError: 500` on ChemGraph but plain `curl` works | Model name is mixed-case (`argo:GPT-4.1-mini`). Use lowercase (`argo:gpt-4.1-mini`) so `supported_argo_models` matches and `argo_user` gets wired. |
| `openai.BadRequestError: 400 - Invalid model: gpt-4.1-mini` | The stripped lowercase-hyphenated form isn't in the shim's model list. Set `export CHEMGRAPH_ARGO_MODEL_FORMAT=wire` so ChemGraph sends the wire form (`gpt41mini`), which the shim accepts. `test_run.py` sets this automatically. |
| Connection refused on `nc -zv <UAN> 18084` | Login-node tunnel died. Restart Step 1a (`argo-shim --tunnel`). Duo push again. |
| Duo push works but SSH says `Permission denied (publickey, password, ...)` | Aurora → CELS auth chain is broken. Confirm `ssh homes.cels.anl.gov` works from the login node manually (`ssh -o BatchMode=yes` for a zero-risk probe). |
| SSH `Too many authentication failures` | Your ssh-agent is offering too many keys. Add `-o IdentitiesOnly=yes -i ~/.ssh/cels_key` to the login-node SSH config for `homes.cels.anl.gov`. |

## Running the test

Once shim health check is green:

```bash
source /path/to/chemgraph/venv/bin/activate
python examples/connecting_to_argo/test_run.py
```

Expected output includes an `INFO` line
`Using OpenAI-style Argo model for local endpoint '...': 'argo:gpt-4.1-mini' -> 'GPT-4.1-mini'`
followed by ChemGraph's answer to the query.
