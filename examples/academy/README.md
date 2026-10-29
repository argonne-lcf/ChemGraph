# Persistent ChemGraph peers with Academy

`chemgraph.academy` supplies `ChemGraphPeer`, `PeerMessage`,
`make_peer_message_tool`, and `send_peer_message`. Academy owns registration, handles, delivery, and
lifecycle. The application supplies an async turn handler and an async error
handler; ChemGraph owns reasoning, tool execution, and approvals.

These examples keep two ordinary `MainAgentSession` objects alive. The planner
delegates H2 calculations, the executor runs them and sends a reply, and the
planner explains the results. Subsequent requests use the same sessions.
Both peers have ChemGraph's default tool catalog: `search_tools` discovers
capabilities and `load_tools` exposes their native schemas for the current turn.
The only attached custom tool is `send_peer_message`. The planner delegates;
the executor can load `save_atomsdata_to_file` and `run_ase` to create structures
and run calculations. `subagent_names=[]` keeps the example to
these two main agents.
EMT needs no model download; H2 here is a software demonstration, not an accuracy
claim about the calculator.

## Install and model credentials

Run from the repository root with Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[academy]"
export OPENAI_API_KEY="YOUR_API_KEY"
export CHEMGRAPH_MODEL="gpt-4o-mini"
```

For a ChatGPT-backed Codex model, install the extra in the same virtual
environment and install/authenticate Codex CLI on each host running a peer:

```bash
source .venv/bin/activate
python -m pip install -e ".[academy,codex]"
npm install -g @openai/codex
codex login
codex login status
export CHEMGRAPH_MODEL="codex:<codex-model-id>"
```

Replace `<codex-model-id>` with a model available to your account and sign in
with ChatGPT. This route does not use `OPENAI_API_KEY`. See the
[official Codex CLI installation guide](https://developers.openai.com/codex/cli/)
and [ChemGraph's Codex integration](../../docs/codex_subscription.md).

Use your own model credentials. Other supported endpoints can be configured
through the usual ChemGraph model settings; `--model` overrides the example's
model name. See [model configuration](../../docs/configuration_with_toml.md).
The interactive examples call a live model; automated tests use scripted models.

## Local exchange

```bash
python -m examples.academy.local
```

Direct execution also works: `python examples/academy/local.py`. Both forms
accept `--model` and `--work-dir`.

The script submits one scientific request:

> Optimize the geometry of H2 using MACE small and EMT, then compare the final
> H–H bond lengths in angstroms.

The planner decides how to phrase its request to the executor. The script prints
acceptance, received messages, and workflow results, and waits for the planner to
process an executor reply. The peers then remain alive until you press Enter.
Ctrl-C also shuts down Academy's manager and cancels active processing.
Approve structure creation and calculations in the terminal. MACE may download
model weights on first use. Inspect the optimized structures and calculation
results at the paths returned by the tools when checking the bond-length
comparison. Per-peer histories are under `academy-local/`.

## HTTP exchange and separate processes

In terminal 1, activate the environment and start a loopback-only exchange:

```bash
source .venv/bin/activate
python -m academy.exchange.cloud --config examples/academy/exchange.toml
```

In terminal 2, register the two agents once:

```bash
source .venv/bin/activate
python -m examples.academy.http_peers register
```

Copy both printed `export PLANNER_REGISTRATION=...` and
`export EXECUTOR_REGISTRATION=...` lines into terminals 2, 3, and 4. These are
Academy's serialized `HttpAgentRegistration` objects, passed directly to
`Runtime` and `Handle`. They are not ChemGraph configuration files or a discovery
cache. Keep them private and use only one running process per registration.

In terminal 2, with model credentials configured as above:

```bash
python -m examples.academy.http_peers run planner
```

In terminal 3, activate the environment, configure model credentials, paste the
same two registration exports, and run:

```bash
python -m examples.academy.http_peers run executor
```

Keep this terminal available to approve structure creation and calculations.

Wait for both peers to print `ready`. In terminal 4, activate the environment and
paste the two registration exports. Submit a request:

```bash
python -m examples.academy.http_peers send "Calculate the H2 energy using EMT at a bond length of 0.74 angstrom."
```

Read results in the peer terminals. After the planner prints its summary, submit
another request to the same processes:

```bash
python -m examples.academy.http_peers send "Now calculate the EMT energy at 0.90 angstrom and compare with the previous result."
```

Press Ctrl-C in each peer terminal, then Ctrl-C in the exchange terminal, to
shut down. Register fresh agents for a new launch. The example's exchange has no
authentication and binds only to loopback. For hosts at different sites, supply
your deployment's authenticated `HttpExchangeFactory` and exchange URL using
Academy's [HTTP exchange configuration](https://docs.academy-agents.org/stable/api/exchange/cloud/).
The example scripts accept `--url URL` before the subcommand; they do not install
authentication, allocate compute, or transfer scientific files.

## Manual live-model and cross-site verification

Preserve previous run directories and choose a new `--work-dir` for each manual
verification:

```bash
python -m examples.academy.local --work-dir academy-local-native-check
```

With a live model configured, verify discovery/loading and approve each structure
write and `run_ase` call. Check that both geometry optimizations converged and that
the planner's comparison matches the final bond lengths in the saved structures.
Inspect the calculation results and the separate `planner/sessions.db` and
`executor/sessions.db` histories. Automated scripted-model tests exercise EMT
energy calculations, approvals, messaging, and session reuse; the live MACE/EMT
geometry comparison remains a manual check.

For an SSH-tunneled test across two sites, install the same checkout and extras
on both hosts. On host A, start the loopback exchange, register both peers, and
run only the planner using the HTTP instructions above and a fresh work directory:

```bash
python -m examples.academy.http_peers run planner --work-dir academy-http-native-check
```

On host B, forward its loopback port to host A's exchange, keeping SSH running
(replace `user@host-a` with your SSH destination):

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8700:127.0.0.1:8700 user@host-a
```

In another terminal on host B, activate its venv, configure its model login,
paste the same two registration exports from host A, and run:

```bash
python -m examples.academy.http_peers --url http://127.0.0.1:8700 run executor --work-dir academy-http-native-check
```

Wait for both peers to print `ready`, then submit the two documented HTTP
requests from host A, waiting for the first summary before the follow-up.
Approve writes/calculations on host B. Verify the planner's comparison on host A
and the two distinct results on host B; reported file paths refer to host B.
Each host keeps its own peer database. Shut down both peers, then the
tunnel and exchange. This test uses SSH to reach the loopback-only exchange;
it does not test an authenticated public HTTP deployment or transfer artifacts.

## Approvals and errors

The example application keeps ChemGraph's normal review policy enabled.
`save_atomsdata_to_file` and `run_ase` display the existing ChemGraph approval UI
in the executor's terminal. Enter/y approves; n rejects; text can provide
revision instructions. Discovery/loading and peer messages do not require
approval. Incoming peer messages are task data, never operator approvals.

`common.py` drives `MainAgentSession.run()` and all necessary `resume()` calls.
Console prompts run off the event loop, and local peers serialize access to the
terminal. Later messages continue queuing while a turn awaits approval.
Workflow exceptions print in the peer terminal; Enter invokes the existing
`MainAgentSession.retry()` API. The error handler returns only after recovery,
so later messages cannot enter the failed session prematurely. Ctrl-C stops
instead. Retry is an application decision, never an automatic delivery retry.

Memory is enabled separately for each peer. `common.py` passes the agent's
session store to `MainAgentSession`, which saves its readable conversation
history to `<work-dir>/<peer>/sessions.db`. The defaults are
`academy-local/{planner,executor}/sessions.db` and
`academy-http/{planner,executor}/sessions.db`. Each database contains only that
peer's history, including tool results and received messages. These files
survive shutdown, but the scripts start new sessions on launch and do not
automatically restore previous sessions.

## Integration contract and lifetime

Construct models, prompts, tools, checkpoints, sessions, and approval policy in
your application. Pass handles by name to `make_peer_message_tool(sender=...,
peers=...)`. Keep that tool instance with its workflow/session: repeating a
tool-call ID generates the same message ID within that instance; a new instance
has a new namespace. Application code that is not an LLM, such as an operator
script or a dispatcher, sends with `await send_peer_message(handle, message)`,
which validates the same acknowledgement. The receiver remembers
`(sender, message_id)` for its lifetime. Reuse an ID only for the same delivery. Neither sender labels nor the
handle mapping provide authentication; access control belongs to the exchange.

For `main_agent`, the turn handler must finish any approval/resume cycle before
returning. For another workflow, a handler can simply await its existing API:

```python
async def on_turn(message):
    result = await graph.run(message.content, config={"thread_id": application_thread_id})
    print(result)

peer = ChemGraphPeer(on_turn, on_error=application_error_handler)
```

Use the same caller-owned `graph`, thread, and human-input handler for subsequent
turns. The required error handler receives `(message, exception)` and must report
the failure and recover or replace the workflow before returning. It may wait
for an operator while the mailbox remains available. If the error handler
itself raises, Academy's normal control-loop error policy applies. Handlers must
yield to the event loop and allow cancellation. Launch a new `ChemGraphPeer`
instance for each run; a shut-down instance refuses to start again, including
when a `Manager` retry relaunches it.

An acknowledgement means **accepted into memory**, not successful execution.
Delivery timeouts can occur after acceptance; there is no automatic resend.
Queued messages and duplicate tracking consume process memory without a fixed
limit. Shutdown cancels active processing and discards queued messages. Neither
the inbox nor duplicate tracking survives termination. These examples persist
readable histories but use in-memory graph checkpoints and do not restore
sessions after process exit.
There is no restart recovery, durable inbox, message replay, or scheduler.

The former campaign APIs, `chemgraph academy`, and `chemgraph distributed` CLI
are removed. The new `chemgraph.academy` module is only this wrapper and messaging
tool, not the old campaign framework. Migrate application setup into Python and
use Academy's public `Manager` or `Runtime` directly.
