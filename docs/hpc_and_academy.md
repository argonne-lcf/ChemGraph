# HPC and Academy

ChemGraph separates agent logic from execution so chemistry tasks can run
locally or through a facility backend. HPC integrations require site modules,
allocations, scheduler policies, endpoint identifiers, credentials, and
shared-filesystem planning beyond a normal PyPI install.

For individual PBS jobs, [Deep Agent can read skills and write batch scripts](pbs_jobs_with_skills.md)
using its submission-host shell and the existing ASE Python engine.

## Execution backends

| Backend | Extra | Typical role |
| --- | --- | --- |
| Local | Core | Validate task packaging and results on one machine |
| Parsl | `parsl` | Submit through a configured Parsl executor |
| Ensemble Launcher | `ensemble_launcher` | Launch ensembles on supported systems |
| Globus Compute | `globus_compute` | Invoke registered remote endpoints |
| Globus Transfer | Core SDK plus credentials | Move inputs/results between collections |

Start with the [execution demos](https://github.com/argonne-lcf/ChemGraph/tree/main/scripts/demo).
They separate direct backend calls from agent-driven variants. Test the direct
path first so infrastructure failures are isolated from LLM behavior.

Backend selection can use environment variables or `[execution]` settings. Do
not copy another user's endpoint IDs, allocation names, or private paths.

## Academy peers

The optional `chemgraph.academy` integration contains a thin `ChemGraphPeer`
Academy agent, a `PeerMessage` payload, and `make_peer_message_tool`. Applications
provide async turn and error handlers and a mapping of names to actual Academy
handles. Use Academy's public `Manager`, `Runtime`, registration, and lifecycle
APIs directly. Computation continues to use the execution backends above.

```bash
uv pip install "chemgraph[academy]"
```

See the [Python examples](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/academy)
for two persistent sessions using local or HTTP exchanges, including installation,
model credentials, submission, approval, and shutdown commands.

The application constructs its model, prompts, tools, sessions, and approval
policy. Its turn handler drives `MainAgentSession.run()` and required `resume()`
calls for `main_agent`, or `ChemGraph.run()` for other workflows. The wrapper
processes one turn at a time and keeps accepting messages while approvals or
application error recovery are pending. It never chooses checkpoint identity,
retries, or replaces workflows itself.

Acknowledgements mean acceptance into an in-memory inbox, not successful
execution. Peers remain alive between messages, but shutdown discards queued
work and duplicate tracking. There is no restart recovery, automatic replay,
durable inbox, or scheduling framework.

### Migration from campaigns and distributed configs

The old campaign APIs, campaign JSONC format, prompt profiles, MPI daemon, and
campaign-specific launchers remain removed. Neither `chemgraph academy` nor
`chemgraph distributed` is a supported command. The new `chemgraph.academy`
module exposes only the wrapper, message payload, and messaging tool; it does
not restore the campaign framework. Move application configuration into Python
and pass Academy registrations/handles explicitly. There is no custom discovery
service, JSON peer configuration, or operator control-file protocol.

Use site job scripts or existing ChemGraph execution backends for HPC placement.
`dev-academy-sim` is a historical prototype, not a required installation.

## ALCF model access and MCP

Compute nodes may not reach the same authentication services as login nodes.
Follow the current [Argo connection example](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/connecting_to_argo)
and ALCF policy for proxies, tokens, outbound networking, and secrets.

ChemGraph also includes ASE, MACE, gRASPA, and XANES HPC MCP modules with direct
or Parsl execution. They are deployment building blocks and can contain
facility-specific assumptions. See [MCP servers](mcp_servers.md).

## Operational checklist

1. Validate the scientific command directly on the target system.
2. Validate the backend without an agent.
3. Confirm input/output visibility across hosts and containers.
4. Run one small agent task with explicit calculator/resource settings.
5. Add concurrency only after logs, retries, and cleanup are understood.
6. Record model, calculator, environment, scheduler, and code versions.

Never place tokens in application configuration files, prompts, repository config, or job logs.
Use facility-approved secret mechanisms and least-privilege endpoints.
