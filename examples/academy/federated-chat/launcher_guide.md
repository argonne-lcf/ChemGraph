# Federated-Chat with the `launch` Subcommand

This is the **launcher-driven** walkthrough: bring up the campaign
across already-running interactive PBS allocations with one local
command. For the four-terminal manual flow, see [`e2e_guide.md`](e2e_guide.md).

The launcher (`chemgraph academy launch`) ssh's from your laptop
into a compute node inside your existing interactive PBS
allocation and runs `chemgraph academy spawn-site` there. The
operator skips the "ssh in, source env, type spawn-site"
ritual for every campaign on every HPC.

## Current support matrix

| HPC | Attach-mode | Submit-mode |
|-----|------------|------------|
| Aurora | yes | unit-tested only |
| Crux | **no** -- see [Crux limitation](#crux-limitation) | unit-tested only |
| Polaris | untested | unit-tested only |

The two-HPC federated-chat demo therefore currently runs
Aurora-only (both agents stacked on Aurora). True cross-HPC
attach is gated on the Crux work below.

## Prerequisites

Same one-time setup as the manual e2e (`e2e_guide.md`): ChemGraph
checkout staged on each HPC, `[academy]` extra installed, Redis
binary built once, hosted-exchange Globus token cached.

Plus on your laptop:
- ChemGraph installed locally
- `~/.ssh/config` has a ControlMaster block for `*.alcf.anl.gov`
  (otherwise every nested ssh re-auths)

## Setup

### 1. Open the interactive PBS allocation

Per HPC. The launcher attaches into an allocation you already
hold open -- it does not qsub for you in attach-mode.

```bash
# Aurora (in a terminal you keep open):
ssh jinchuli@aurora.alcf.anl.gov
qsub -I -A ChemGraph -q debug -l select=2,walltime=01:00:00 \
     -l filesystems=home:flare
# When the prompt lands on a compute node:
echo "AURORA_COMPUTE=$(hostname)"
echo "AURORA_JOBID=$PBS_JOBID"
# Copy these two values to your laptop shell.
```

### 2. Export everything on your laptop

```bash
export ALCF_PROJECT=ChemGraph
export ALCF_USER=jinchu               # path component on /flare
export ALCF_SSH_USER=jinchuli         # ssh login (may differ from above)
export ARGO_USER=jinchu.li            # Argo / email-prefix
export AURORA_COMPUTE=<from step 1>
export AURORA_JOBID=<from step 1>
```

**The three look-alike usernames:**
- `jinchu` -> ALCF_USER, the workspace path under `/flare/...`
- `jinchuli` -> ALCF_SSH_USER, the ssh login
- `jinchu.li` -> ARGO_USER, the LM API identifier

The launcher reads these from your laptop shell and forwards them
to the remote bash so spawn-site sees the same values it does in
your manual qsub workflow.

### 3. Start the dashboard

The dashboard launcher brings up the **UAN HTTP relay** on each
named HPC (so daemons can reach Argo for LM calls), opens a
ControlMaster ssh, and rsync-mirrors the run dir to your laptop.
Without it, the spawn-site daemon will fail at LM-config
resolution because no relay-host file exists.

```bash
chemgraph academy dashboard -- federated-launch-001 \
  --system aurora \
  --campaign federated-chat
```

Leave this running in its own terminal. Wait until it prints
`Aurora relay host: uan-XXXX` before continuing.

### 4. Run the launcher

```bash
chemgraph academy launch -- \
  --run-id federated-launch-001 \
  --campaign federated-chat \
  --bundle-root /flare/${ALCF_PROJECT}/${ALCF_USER}/ChemGraph \
  --site "aurora:attach=${AURORA_COMPUTE};agents=agent-aurora,agent-crux;pbs_jobid=${AURORA_JOBID}" \
  --auto-bootstrap \
  --spawn-arg=--agents-per-node --spawn-arg=2
```

What should print:

```text
[launch] sites=[aurora(attach)] run-id=federated-launch-001
[attach:aurora] waiting for ['agent-aurora', 'agent-crux'] to register (elapsed 30s of 300s)
[launch] ready: aurora -> ['agent-aurora', 'agent-crux']
[launch] all sites ready, dispatching bootstrap...
[launch] bootstrap dispatched.
[launch] launch complete. Compute processes continue running.
```

The agents now exchange the counter ping-pong; watch the dashboard
or tail `events.jsonl` on the compute side. The campaign self-
terminates when the counter hits 10.

## `--site` flag reference

```text
--site NAME:KEY=VAL;KEY=VAL;...
```

Pairs are separated by `;` so comma stays free for the `agents=`
CSV. Mix and match multiple `--site` flags for multi-HPC runs.

Common (both modes):
- `agents=<csv>` -- required. Which agents this site runs.
- `bundle_root=<path>` -- per-site override of the global
  `--bundle-root`. Needed when HPCs use different filesystems
  (Aurora `/flare`, Crux `/eagle`).

Attach-mode (presence of `attach=`):
- `attach=<compute_host>` -- required. From `hostname` inside the
  operator's `qsub -I`.
- `pbs_jobid=<id>` -- strongly recommended for multi-node runs.
  Without this, the launcher's nested ssh strips PBS env, mpiexec
  defaults to current-host-only, and any agent_count > 1 fails
  with `Cannot place all ranks on node list`. Copy from `echo
  $PBS_JOBID` inside the operator's qsub shell.

Submit-mode (presence of `queue=`):
- `queue=<name>` -- required.
- `walltime=<HH:MM:SS>` -- required.
- `nodes=<int>` -- optional, default 1.
- `project=<alloc>` -- optional, falls back to `--project`.
- `filesystems=<csv>` -- optional, default per system profile.

## Useful extra flags

- `--spawn-arg=K --spawn-arg=V`: append `K V` to the remote
  spawn-site argv. Use `=` syntax so argparse doesn't try to
  parse `K` as its own flag. Useful for things the launcher
  doesn't expose a dedicated CLI for (`--agents-per-node`,
  `--max-decisions`, etc.).
- `--ready-timeout-s 600`: how long to wait for agents to
  register, post-allocation. Default 300s. Bump for slower
  startups or busy queues.
- `--exchange-type http`: default. Use `redis` / `local` /
  `hybrid` for single-machine or single-allocation runs.

## What happens on Ctrl-C

The launcher signals SIGTERM to its outer ssh, which `-tt`
forwards as SIGHUP to the remote python -- agents die cleanly
inside your allocation. The PBS allocation itself stays up (it's
your interactive shell, not the launcher's). Re-run the launcher
to resume.

## Failure surfacing

If `wait_ready` raises or the ssh chain dies before agents
register, the launcher prints the last 30 lines of the per-site
`attach.log`. This is the truth -- always read it first. It lives
at:

```text
<remote run_dir>/<site>.attach.log
```

For Aurora that's `/flare/${ALCF_PROJECT}/${ALCF_USER}/runs/<run-id>/aurora.attach.log`.

You can also tail it from your laptop while the run is live:

```bash
ssh jinchuli@aurora.alcf.anl.gov \
  "tail -f /flare/${ALCF_PROJECT}/${ALCF_USER}/runs/<run-id>/aurora.attach.log"
```

## Crux limitation

Crux compute nodes don't trust the login node for hostbased /
passwordless ssh. Every `ssh COMPUTE` from a Crux login node
prompts for a password, which kills the launcher's non-interactive
ssh chain. Aurora is unaffected (its compute nodes accept
hostbased auth).

For Crux today, use the **four-terminal manual flow** in
[`e2e_guide.md`](e2e_guide.md). The launcher still handles
Aurora-only runs from one command.

Tracked options:
1. Add `pbsdsh` as an alternate attach transport for Crux.
2. Use submit-mode for Crux (qsub via the launcher each campaign,
   no attach to existing allocation).
3. Defer until cross-HPC Aurora<->Crux is a real ask.
