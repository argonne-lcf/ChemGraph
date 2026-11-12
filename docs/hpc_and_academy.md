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

## Academy campaigns

Academy supports persistent multi-agent campaigns and a dashboard runtime.
Install it into the environment ChemGraph runs in:

```bash
uv pip install "chemgraph[academy,parsl,globus_compute]"
# or, in a conda/pip environment: python -m pip install "chemgraph[academy,parsl,globus_compute]"
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

## PBS and machine guidance

Compose operational guidance as:

```text
PBS operations + selected machine guidance + software guidance
```

The packaged [pbs skill](../src/chemgraph/skills/pbs/SKILL.md) provides portable
scheduler operations through native shell commands. It works independently of
ChemGraph imports and can be copied into another compatible agent's skill
directory.

The [pbs-hpc skill](../src/chemgraph/skills/pbs-hpc/SKILL.md) provides ChemGraph
batch examples and machine references for Aurora, Polaris, Crux, and
[Bebop](../src/chemgraph/skills/pbs-hpc/references/bebop.md). Bebop is a reference
loaded through `pbs-hpc`, not a separate skill. It covers resources, queues,
accounts, storage, modules, and dated official sources.

Example prompts:

- “Use the PBS skill to inspect this job on the current machine.”
- “Use pbs-hpc and its Bebop reference to prepare this submission.”
- “Use the PBS skill with the supplied Polaris settings.”

Existing execution configurations and Academy profiles are unchanged. Software
skills, including Gaussian, are future work.

`chemgraph.skills.skill_directory()` locates the installed skill root. Existing
bundled-skill discovery exposes `pbs` and `pbs-hpc`; see
[skill discovery](skills.md). Native PBS IDs must not be passed to ChemGraph's
Future-based backend job tools.
