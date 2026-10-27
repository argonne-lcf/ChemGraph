# Persistent ChemGraph peers with Academy

`chemgraph.academy` supplies `ChemGraphPeer`, `PeerMessage`, and
`make_peer_message_tool`. Academy owns registration, handles, delivery, and
lifecycle. The application supplies an async turn handler and an async error
handler; ChemGraph owns reasoning, tool execution, and approvals.

These examples keep two ordinary `MainAgentSession` objects alive. The planner
delegates an H2 calculation, the executor uses ASE's EMT calculator and sends a
reply, and the planner summarizes it. Subsequent requests use the same sessions.
EMT needs no model download; H2 here is a software demonstration, not an accuracy
claim about the calculator.

## Install and model credentials

Run from the repository root with Python 3.11 or newer:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[academy]"
export OPENAI_API_KEY="YOUR_API_KEY"
export CHEMGRAPH_MODEL="gpt-4o-mini"
```

Use your own model credentials. Other supported endpoints can be configured
through the usual ChemGraph model settings; `--model` overrides the example's
model name. See [model configuration](../../docs/configuration_with_toml.md).
The interactive examples call a live model; automated tests use scripted models.

## Local exchange

```bash
python -m examples.academy.local
```

The script submits two requests in sequence, at 0.74 and 0.90 angstrom, and waits
for each reply. It prints acceptance, received messages, and workflow results.
After both requests, the peers remain alive until you press Enter. Ctrl-C also
shuts down Academy's manager and cancels active processing.

## HTTP exchange and separate processes

In terminal 1, activate the environment and start a loopback-only exchange:

```bash
source .venv/bin/activate
python -m academy.exchange.cloud --config examples/academy/exchange.toml
```

In terminal 2, register the two agents once:

```bash
source .venv/bin/activate
python -m examples.academy.http register
```

Copy both printed `export PLANNER_REGISTRATION=...` and
`export EXECUTOR_REGISTRATION=...` lines into terminals 2, 3, and 4. These are
Academy's serialized `HttpAgentRegistration` objects, passed directly to
`Runtime` and `Handle`. They are not ChemGraph configuration files or a discovery
cache. Keep them private and use only one running process per registration.

In terminal 2, with model credentials configured as above:

```bash
python -m examples.academy.http run planner
```

In terminal 3, activate the environment, configure model credentials, paste the
same two registration exports, and run:

```bash
python -m examples.academy.http run executor
```

Wait for both peers to print `ready`. In terminal 4, activate the environment and
paste the two registration exports. Submit a request:

```bash
python -m examples.academy.http send "Ask executor for H2 EMT energy at 0.74 angstrom."
```

Read results in the peer terminals. After the planner prints its summary, submit
another request to the same processes:

```bash
python -m examples.academy.http send "Now ask executor for H2 EMT energy at 0.90 angstrom and compare with the previous result."
```

Press Ctrl-C in each peer terminal, then Ctrl-C in the exchange terminal, to
shut down. Register fresh agents for a new launch. The example's exchange has no
authentication and binds only to loopback. For hosts at different sites, supply
your deployment's authenticated `HttpExchangeFactory` and exchange URL using
Academy's [HTTP exchange configuration](https://docs.academy-agents.org/stable/api/exchange/cloud/).
The example scripts accept `--url URL` before the subcommand; they do not install
authentication, allocate compute, or transfer scientific files.

## Approvals, errors, and tracing

The example application keeps ChemGraph's normal review policy enabled. File
mutations and other policy-covered operations display the existing ChemGraph
approval UI in the receiving peer's terminal. Enter/y approves; n rejects; text
can provide revision instructions. The simple EMT and message tools are not
covered by the default review policy. For an explicit approval demonstration:

```bash
python -m examples.academy.http send "Write a short note about the last result to result-note.txt."
```

`common.py` drives `MainAgentSession.run()` and all necessary `resume()` calls.
Console prompts run off the event loop, and local peers serialize access to the
terminal. Later messages continue queuing while a turn awaits approval.
Workflow exceptions print in the peer terminal; Enter invokes the existing
`MainAgentSession.retry()` API. The error handler returns only after recovery,
so later messages cannot enter the failed session prematurely. Ctrl-C stops
instead. Retry is an application decision, never an automatic delivery retry.

Ordinary ChemGraph tracing is available per workflow:

```bash
chemgraph dashboard -- --run-dir academy-local/planner/trace
# Or, for the HTTP example:
chemgraph dashboard -- --run-dir academy-http/executor/trace
```

There is no peer-specific dashboard or shared remote status service. The shared
dashboard and tracing modules also work without Academy installed.

## Integration contract and lifetime

Construct models, prompts, tools, checkpoints, sessions, and approval policy in
your application. Pass handles by name to `make_peer_message_tool(sender=...,
peers=...)`. Keep that tool instance with its workflow/session: repeating a
tool-call ID generates the same message ID within that instance; a new instance
has a new namespace. The receiver remembers `(sender, message_id)` for its
lifetime. Reuse an ID only for the same delivery. Neither sender labels nor the
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
yield to the event loop and allow cancellation.

An acknowledgement means **accepted into memory**, not successful execution.
Delivery timeouts can occur after acceptance; there is no automatic resend.
Queued messages and duplicate tracking consume process memory without a fixed
limit. Shutdown cancels active processing and discards queued messages. Neither
the inbox nor duplicate tracking survives termination. These examples use
in-memory graph checkpoints and do not restore sessions after process exit.
There is no restart recovery, durable inbox, message replay, or scheduler.

The former campaign APIs, `chemgraph academy`, and `chemgraph distributed` CLI
are removed. The new `chemgraph.academy` module is only this wrapper and messaging
tool, not the old campaign framework. Migrate application setup into Python and
use Academy's public `Manager` or `Runtime` directly.
