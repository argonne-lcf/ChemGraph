# Independent ChemGraph peers

Academy provides registration, discovery, and message transport. Each peer runs
an ordinary ChemGraph graph and owns its tools, memory, approvals, and lifecycle.
The CLI launcher targets POSIX hosts (Linux/macOS), including HPC job allocations.

## Two peers on one host

Install `chemgraph[academy]` and configure model credentials using the normal
ChemGraph environment variables. The examples use `gpt-4o-mini`; change each
peer's `model_name` and optional `base_url` for your provider.

For a local demonstration, start Academy's HTTP exchange in its own terminal:

```bash
python -m academy.exchange.cloud --config examples/distributed/exchange.toml
```

The example exchange binds to localhost. For production, use a separately
managed exchange with the authentication and network access your sites require.
The ChemGraph adapter uses Academy's public client APIs only.

Start the scientific MCP server in another terminal:

```bash
python -m chemgraph.mcp.mcp_tools --transport streamable_http --host 127.0.0.1 --port 9003
```

Choose a new `launch_id` in `two-peers.json` for each independent run. All peers
and operator clients participating in that run must use the same run and launch
IDs. Start each peer in a separate terminal:

```bash
chemgraph distributed run --config examples/distributed/two-peers.json --peer executor
chemgraph distributed run --config examples/distributed/two-peers.json --peer planner
```

Once both report ready, submit work once:

```bash
chemgraph distributed send --config examples/distributed/two-peers.json --peer planner \
  --message 'Ask executor to generate an H2 molecule and calculate its single-point potential energy with the EMT calculator. Summarize the result.'
chemgraph distributed status --config examples/distributed/two-peers.json --peer planner
```

A message acknowledgement means acceptance into the peer's **in-memory inbox**.
It does not mean that a calculation succeeded. Read peer status and event logs
for results. A send timeout can occur after acceptance; inspect status before
resubmitting. The runtime does not automatically replay messages or promise
exactly-once execution after a crash.

## Main-agent and multi-agent graphs

Use `main-agent.json` with the same commands to run a checkpointed `main_agent`
planner and `multi_agent` executor. Main-agent checkpoints and transcripts live
under `run_dir/run_id/launch_id/peer/`, separately from the peer's artifacts.
Restarting with the same configuration restores main-agent conversation state;
it does not restore queued peer messages. Changes to configuration require a
new launch ID.

Approvals are unchanged. When status reports `waiting_for_user`, inspect its
`interrupts`, then submit an explicit operator response **on that peer's host**:

```bash
chemgraph distributed resume --config examples/distributed/main-agent.json --peer planner \
  --response '{"decisions":[{"type":"reject"}]}'
```

Use `approve` instead of `reject` only when approving the displayed tool action.
For an ordinary clarification, pass a JSON string such as `--response '"EMT"'`.
Multiple interrupts use the existing ChemGraph mapping of interrupt IDs to
responses. Operator requests are local files tied to the current process and
pending message; the command prints where its result will appear. Peer messages
cannot resume approvals. Failed main-agent turns support the local `retry`
command, which resumes the checkpoint without appending the input again.

## Two systems

Copy the peer configuration to both systems and set `exchange_url` to a shared,
reachable HTTP exchange. Use Academy's supported authentication for that
exchange; the hosted default URL uses Academy's Globus authentication. Each
host may use its own `run_dir`. Configure the executor's MCP URL from the
executor host's perspective. Start each process through that site's ordinary
shell or job script; Academy does not schedule jobs or launch simulations.

MCP servers and execution backends own scientific files. A peer message does
not make a path on another host readable. Transfer inputs/results using the
existing ChemGraph transfer tools, or send small textual summaries.

## Lifecycle and inspection

Peers remain available after a final answer. `idle_timeout_s` is disabled by
default; when configured, it applies only with no active or queued work.

```bash
chemgraph distributed shutdown --config examples/distributed/two-peers.json --peer planner
chemgraph distributed shutdown --config examples/distributed/two-peers.json --peer executor
chemgraph dashboard -- --run-dir examples/distributed/runs/peer-demo/LAUNCH_ID/planner
```

Shutdown commands run locally on the corresponding peer host. SIGINT/SIGTERM
also request shutdown. The runtime stops accepting messages and drains queued
work up to `timeout_s`; it records a shutdown timeout if work remains, then
cancels local processing. Approval-blocked or failed turns reach that timeout
unless an operator resolves them before shutdown.

`status.json` and `events.jsonl` provide per-peer inspection. No shared status
files, global campaign controller, or Academy-specific reasoning loop is used.
