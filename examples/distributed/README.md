# Independent ChemGraph peers

Academy provides registration, discovery, and message transport. Each peer runs
an ordinary ChemGraph graph and owns its tools, memory, approvals, and lifecycle.
Distributed peers default to `main_agent`, driven by `MainAgentSession` with
per-peer SQLite checkpoints and conversation storage.
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

The primary example, `main-agent.json`, runs both planner and executor as
`main_agent`. Choose a new `launch_id` for each independent run. All peers
and operator clients participating in that run must use the same run and launch
IDs. Start each peer in a separate terminal:

```bash
chemgraph distributed run --config examples/distributed/main-agent.json --peer executor
chemgraph distributed run --config examples/distributed/main-agent.json --peer planner
```

Once both report ready, submit work once:

```bash
chemgraph distributed send --config examples/distributed/main-agent.json --peer planner \
  --message 'Ask executor to generate an H2 molecule and calculate its single-point potential energy with the EMT calculator. Summarize the result.'
chemgraph distributed status --config examples/distributed/main-agent.json --peer planner
chemgraph distributed status --config examples/distributed/main-agent.json --peer executor
```

A message acknowledgement means acceptance into the peer's **in-memory inbox**.
It does not mean that a calculation succeeded. Read peer status and event logs
for results. A send timeout can occur after acceptance; inspect status before
resubmitting. The runtime does not automatically replay messages or promise
exactly-once execution after a crash. The planner's final summary appears in the
`output` field of its status and in its event log. `operator` identifies the
human caller; agents report to it with a normal final response, not a peer message.

## Capabilities and approvals

Each main-agent peer receives its configured MCP tools and allowed peer messaging,
plus the standard main-agent workspace features. The local science-tool catalog
is empty, skill discovery defaults to `false`, and local specialist selection
defaults to `[]`. This keeps scientific execution on the configured MCP server.
Explicit skill sources and named specialists remain available through the existing
`options.skills`, `options.skill_dirs`, and `options.subagent_names` settings;
`options.discover_skills: true` explicitly enables skill discovery. Opting into
a specialist grants that specialist's normal capabilities. Use `prompts.main_agent`
for a main-agent peer's role instructions.

Normal ChemGraph approval policy remains enabled. In this example, the executor
pauses before reviewed scientific operations such as `smiles_to_coordinate_file`
and `run_ase`. When its status reports `waiting_for_user`, inspect its `interrupts`,
then submit an explicit operator response **on that peer's host**:

```bash
chemgraph distributed resume --config examples/distributed/main-agent.json --peer executor \
  --response '{"decisions":[{"type":"approve"}]}'
```

Use `approve` only after reviewing the displayed tool action; use `reject` to decline.
Include one decision per displayed action and check status again after each resume,
since the next tool operation may require another approval. The planner may remain
idle while waiting for the executor's result.
For an ordinary clarification, pass a JSON string such as `--response '"EMT"'`.
Multiple interrupts use the existing ChemGraph mapping of interrupt IDs to
responses. Operator requests are local files tied to the current process and
pending message; the command prints where its result will appear. Peer messages
cannot resume approvals. Failed main-agent turns support the local `retry`
command, which resumes the checkpoint without appending the input again.

## Persistence and legacy workflows

Main-agent checkpoints and transcripts live under `run_dir/run_id/launch_id/peer/`,
separately from the peer's artifacts. Restarting with the same configuration restores
main-agent conversation state; it does not restore queued peer messages. Changes to
workflow or capabilities require a new launch ID. Existing checkpoints are not converted.

`two-peers.json` remains a legacy single-agent example. Explicit `single_agent` and
`multi_agent` selections still work, with graph checkpoints held in memory. Configs
that omitted `workflow_type` now select `main_agent`; set the legacy workflow explicitly
to retain the previous behavior. ChemGraph's global workflow default is unchanged.

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
chemgraph distributed shutdown --config examples/distributed/main-agent.json --peer planner
chemgraph distributed shutdown --config examples/distributed/main-agent.json --peer executor
chemgraph dashboard -- --run-dir examples/distributed/runs/main-agent-demo/LAUNCH_ID/planner
```

Shutdown commands run locally on the corresponding peer host. SIGINT/SIGTERM
also request shutdown. The runtime stops accepting messages and drains queued
work up to `timeout_s`; it records a shutdown timeout if work remains, then
cancels local processing. Approval-blocked or failed turns reach that timeout
unless an operator resolves them before shutdown.

`status.json` and `events.jsonl` provide per-peer inspection. No shared status
files, global campaign controller, or Academy-specific reasoning loop is used.
