# Federated-Chat E2E Guide

This guide runs the `federated-chat` ChemGraph Academy campaign across
**two HPCs simultaneously** (Aurora and Crux), with the dashboard on
your laptop merging both sites into one view.

The campaign is intentionally minimal: two agents bounce a counter back
and forth across the HPC boundary, each incrementing it, until it hits 10.
It exercises every part of the cross-HPC stack (deterministic peer
discovery, HTTP exchange, cross-site send_message, multi-site dashboard)
without needing any science tools.

```text
agent-aurora           agent-crux
   ↓ counter=1 ──►        receive
   receive ◄── counter=2  ↓
   ↓ counter=3 ──►        receive
   ...                    ...
   receive ◄── counter=10  ↓
   finish_turn            finish_turn
```

Four terminals: dashboard (Mac), Aurora compute, Crux compute, bootstrap (Mac).

## Configure Paths

Set these in every shell (Mac + both HPCs):

```bash
export ALCF_PROJECT=ChemGraph
export ALCF_USER=<shared-filesystem-user>      # e.g. jinchu
export ALCF_SSH_USER=<ssh-login>               # may differ, e.g. jinchuli
export ARGO_USER=<argo-user>                   # e.g. jinchu.li
export LOCAL_CHEMGRAPH=<local-chemgraph-checkout>
```

`ALCF_USER` is the shared-filesystem path component (`/flare/$ALCF_PROJECT/$ALCF_USER`).
`ALCF_SSH_USER` is the SSH login. They may differ; the loader defaults
`ALCF_SSH_USER` to `ALCF_USER` if you don't set it.

## One-Time Setup

You need the same setup as `example-002-mace-ensemble-screening` (sync
ChemGraph, install `[academy]` extra, build Redis once) on **both** Aurora
and Crux. Plus one extra step: log in to Academy's hosted exchange so the
Globus token is cached on both compute environments:

```bash
# On Aurora compute (inside an interactive allocation):
python -c "from academy.exchange.cloud import HttpExchangeFactory; HttpExchangeFactory()"
# Follow the device-flow URL printed in the terminal. Same on Crux.
```

The token is written to `~/.local/share/academy/storage.db` and is
shared across runs.

## Terminal 1: Dashboard (Mac)

```bash
cd "$LOCAL_CHEMGRAPH"

export RUN_ID=federated-chat-001

chemgraph academy dashboard -- "$RUN_ID" \
  --system aurora,crux \
  --campaign federated-chat \
  --reverse-port 18190 \
  --overwrite-run
```

This brings up:

- one SSH ControlMaster + UAN relay + rsync mirror **per site** (`aurora` and `crux`),
- a single merged dashboard server at `http://127.0.0.1:8765`.

Wait for both relays to print `... relay ready at ...` before continuing.

## Terminal 2: Aurora compute (inside Aurora PBS allocation)

```bash
module load frameworks
source /flare/$ALCF_PROJECT/$ALCF_USER/venvs/academy-swarm/bin/activate
export PATH=/flare/$ALCF_PROJECT/$ALCF_USER/bin:$PATH

# HTTP exchange must reach exchange.academy-agents.org through the ALCF proxy.
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy=127.0.0.1,localhost,.alcf.anl.gov
export NO_PROXY=$no_proxy

chemgraph academy spawn-site -- \
  --system aurora \
  --run-id "$RUN_ID" \
  --campaign federated-chat \
  --agents agent-aurora \
  --exchange-type http
```

Look for the lifecycle landmarks:

```text
[daemon] rank0 registered 'agent-aurora' on the exchange (uid=...)
[daemon] rank0 waiting for peers ['agent-crux'] to come online (timeout 600s)...
[daemon] rank0 all 1 peer(s) are alive: ['agent-crux']
[daemon] rank0 agent 'agent-aurora' is now running inside Academy Runtime
[daemon] rank0 skipping inline bootstrap (federated mode); waiting for 'chemgraph academy bootstrap'...
```

## Terminal 3: Crux compute (inside Crux PBS allocation)

```bash
source /eagle/$ALCF_PROJECT/$ALCF_USER/venvs/academy-swarm-crux/bin/activate
export PATH=/eagle/$ALCF_PROJECT/$ALCF_USER/bin:$PATH

export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
export HTTP_PROXY=$http_proxy
export HTTPS_PROXY=$https_proxy
export no_proxy=127.0.0.1,localhost,.alcf.anl.gov
export NO_PROXY=$no_proxy

chemgraph academy spawn-site -- \
  --system crux \
  --run-id "$RUN_ID" \
  --campaign federated-chat \
  --agents agent-crux \
  --exchange-type http
```

Same landmarks. Both daemons will block at `waiting for 'chemgraph academy
bootstrap'` once they've discovered each other.

## Terminal 4: Bootstrap kickoff (Mac, once both sites are waiting)

```bash
chemgraph academy bootstrap -- \
  --campaign federated-chat \
  --run-id "$RUN_ID" \
  --exchange-type http
```

Prints `ok: sent bootstrap to agent-aurora (message_id=...)`.

## What You Should See

- **Aurora terminal**: `[agent agent-aurora] first message arrived from
  'campaign' ...`, then decisions firing, then `message_sent` to agent-crux.
- **Crux terminal**: `[agent agent-crux] first message arrived from
  'agent-aurora' ...`, then back-and-forth.
- **Dashboard**: agent nodes appear in the graph, metrics tick up, the
  cross-site message-flow edge between aurora and crux fills in, counter
  messages climb in the activity log from 1 → 10.

## Troubleshooting

**Both sides stuck at `waiting for peers` past ~60s** → one site isn't
actually registered. Check each compute terminal for the `registered` line.
If one is missing, the daemon hit an exception before registration; scroll
up.

**`Address already in use` on relay startup** → a prior crashed launch
left an orphan. The new self-cleaning relay should handle it
automatically; if it doesn't, the local relay log under
`/tmp/chemgraph-academy-<run-id>-<site>-relay.log` will have a full `set
-x` trace showing exactly which step failed.

**Bootstrap times out** → both sites must already be at `waiting for
'chemgraph academy bootstrap'`. If only one is up, bootstrap can't find
the recipient.

**Argo `<argo-user>` error** → you didn't export `ARGO_USER` before
launching spawn-site. The launcher refuses to ship a config with the
template placeholder; the error message names the fix.

**`Could not validate Globus token`** → the device-flow login expired.
Re-run the `python -c "from academy.exchange.cloud ..."` snippet from
the one-time setup section.
