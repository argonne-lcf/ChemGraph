# HPC and Academy

ChemGraph separates agent logic from execution so chemistry tasks can run
locally or through a facility backend. HPC integrations require site modules,
allocations, scheduler policies, endpoint identifiers, credentials, and
shared-filesystem planning beyond a normal PyPI install.

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

## Public ALCF Transfer profiles

ChemGraph keeps backend-neutral facility metadata in
`chemgraph.hpc_configs.profiles`. Globus collection UUIDs are public
identifiers, not credentials: Transfer still enforces login, consent, and
collection ACLs. User collection IDs, project names, credentials, and private
paths must stay in environment variables or local configuration.

| System | Collection | Bundled ID | Transfer path | Compute path |
| --- | --- | --- | --- | --- |
| Polaris | `alcf#dtn_eagle` | `05d2c76a-e867-4f67-aa57-76edeb0beda0` | `/<project>/...` | `/eagle/<project>/...` |
| Aurora | `alcf#dtn_flare` | `f39a7a0f-5bfc-46ce-9615-ba9f8592814f` | `/<project>/...` | `/flare/<project>/...` |

Set `COMPUTE_SYSTEM=polaris` or `COMPUTE_SYSTEM=aurora` (or the equivalent
`[execution] system` value) and omit
`GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID` to select the bundled destination.
The profiles translate collection-visible paths to the paths seen by compute
workers: Eagle projects gain the `/eagle` prefix on Polaris, while Flare
projects gain the `/flare` prefix on Aurora.

Explicit arguments take priority, followed by `[execution.globus_transfer]`
settings and environment fallbacks. A custom Polaris collection disables the
automatic path translation unless
`GLOBUS_TRANSFER_DESTINATION_COMPUTE_BASE_PATH` (or
`destination_compute_base_path` in TOML) is also set.

```toml
[execution]
system = "polaris"

[execution.globus_transfer]
source_endpoint_id = "<your-source-collection-uuid>"
destination_base_path = "/<project>/staging"
```

HPC MCP servers always expose `list_transfer_facilities`. Agents can use it to
see both profiles and identify the active server target. Selection is fixed at
server startup so the Transfer destination stays aligned with the configured
Compute endpoint.

See the ALCF documentation for current
[Eagle and Flare collection paths](https://docs.alcf.anl.gov/data-management/data-transfer/using-globus/)
and [Aurora path mapping](https://docs.alcf.anl.gov/aurora/data-management/moving_data_to_aurora/globus/).

## Academy campaigns

Academy supports persistent multi-agent campaigns and a dashboard runtime:

```bash
python -m pip install "chemgraph[academy,parsl,globus_compute]"
chemgraph academy --help
chemgraph dashboard --help
```

The [MACE ensemble-screening example](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/academy/example-002-mace-ensemble-screening)
has a README and end-to-end guide. Review campaign JSON/JSONC, runtime profiles,
prompt profiles, and data paths before launch.

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

Never place tokens in campaign files, prompts, repository config, or job logs.
Use facility-approved secret mechanisms and least-privilege endpoints.
