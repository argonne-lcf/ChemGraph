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

## Distributed peers

Academy is an optional transport for messages between independent ChemGraph
instances. ChemGraph owns reasoning, tools, approval handling, sessions, inbox
processing, and shutdown. Computation continues to use the execution backends
above; the messaging adapter does not allocate compute resources.

```bash
python -m pip install "chemgraph[academy]"
chemgraph distributed --help
chemgraph dashboard --help
```

See the [distributed-peer examples](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/distributed)
for two single-agent peers, a checkpointed main-agent peer with a multi-agent
executor, and deployment across hosts. Each host needs access to the same HTTP
exchange, but peers do not need a shared filesystem.

### Migration from campaigns

The old `chemgraph academy` commands, `chemgraph.academy` Python package,
campaign JSONC format, prompt profiles, MPI daemon, and campaign-specific
launchers have been removed. Configure ordinary ChemGraph peers and launch them
with `chemgraph distributed run --config peers.json --peer NAME`. Use site job
scripts or existing ChemGraph execution backends for HPC placement.

The `academy` extra remains available for the communication adapter. There is
no automatic campaign conversion. Re-express missions as peer prompts, tool
access as per-server MCP allowlists, and connectivity as `allowed_peers`.
`dev-academy-sim` is a historical prototype, not a required installation.

`chemgraph dashboard` still reads existing event logs, including old campaign
runs. Dashboard and CLI tracing work without Academy installed.

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

Never place tokens in peer configuration files, prompts, repository config, or job logs.
Use facility-approved secret mechanisms and least-privilege endpoints.
