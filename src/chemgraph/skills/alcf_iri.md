---
name: alcf_iri
description: Query and manage ALCF HPC resources -- machine status, project allocations, PBS jobs, filesystem contents -- via ALCF's IRI Facility API. Reach for this whenever a user asks about ALCF machines (Aurora, Crux, Polaris, Sophia), their own ALCF projects/allocations, jobs they've submitted, or files on ALCF filesystems.
mcp_server: chemgraph.mcp.alcf_iri_mcp
tool_prefix: alcf_
---

## When to use

Trigger this skill when the user's ask involves any of:

- **ALCF machine state** -- "Is Aurora up? Which machines are in maintenance?"
- **Incidents / events / outages** -- "Any recent Polaris outages? Was there scheduled maintenance last Tuesday?"
- **Projects and allocations** -- "What projects am I on? How much time do I have left on Crux?"
- **PBS jobs** -- "Show my last completed job on Polaris. What's the status of job 246115?"
- **Job submission** -- "Submit a smoke-test job on Crux." *(requires `$ALCF_IRI_ALLOW_UNSAFE=1`)*
- **ALCF filesystem** -- "List /eagle/<project>. Read the output file from my last job." *(requires ALCF-side identity mapping)*

Do NOT reach for this skill for:
- General HPC questions unrelated to ALCF specifically (use the compute skill).
- Reading local files or non-ALCF remote paths (use the filesystem skill).
- Chemistry calculations, even if they will eventually run on ALCF (use the calculator/MACE/gRASPA skills; only invoke this one for the surrounding job-management steps).

## Auth

Public endpoints (facility metadata, machine status, incidents, events) work with no auth. Everything else needs a Globus-issued ALCF Facility API token, resolved in this order:

1. `$ALCF_API_TOKEN` env var (if set).
2. On-disk Globus refresh cache at `~/.globus/app/8b84fc2d-.../alcf_facility_api_app/tokens.json` -- silently refreshed on demand for 30 days after first auth.
3. Interactive re-auth: call `alcf_auth_start_reauth` to get a Globus URL, have the user paste back the code, then call `alcf_auth_complete_reauth(auth_code=...)`.

**Identity gotcha:** users often have both `<user>@anl.gov` (Argonne primary) and `<user>@alcf.anl.gov` (ALCF-linked) identities in Globus. Only the second grants IRI access. When re-auth fails with *"authenticated as ... anl.gov, use alcf.anl.gov"*, tell the user to visit https://app.globus.org/logout and re-run the flow, explicitly picking the `@alcf.anl.gov` identity on the sign-in page.

## Safety

Write actions (`alcf_compute_submit_job`, `alcf_compute_cancel_job`, `alcf_filesystem_mkdir`, `alcf_filesystem_rm`, `alcf_filesystem_chmod`, ...) require `$ALCF_IRI_ALLOW_UNSAFE=1` in the MCP server's environment and will raise `RuntimeError` otherwise. Attempt them normally when the user asks -- do NOT refuse preemptively; the tool layer enforces the gate.

## Tool naming

One tool per API action, named `alcf_<category>_<action>`. Categories are:

| Category    | Tools | Auth  | Typical use                                     |
|-------------|------:|-------|-------------------------------------------------|
| facility    |     3 | none  | Site metadata                                   |
| status      |     6 | none  | Machine state, incidents, events                |
| account     |     7 | token | Projects, allocations, per-user allocation slices |
| compute     |     5 | token | List/submit/status/cancel PBS jobs              |
| filesystem  |    17 | token | ls, stat, read, mkdir, rm, chmod, ...           |
| task        |     3 | token | Async task tracking (long-running IRI ops)      |
| auth        |     2 | none  | Interactive Globus re-auth (in-process helpers) |

Total: 43 tools. Naming is stable -- `alcf_compute_list_jobs`, `alcf_account_list_allocations`, etc. Pick by the action you want; every tool's argument schema is self-describing.

## Example flows

**Machine status (one hop, no auth):**
> User: "Which ALCF machines are currently up?"
> Call `alcf_status_list_resources(current_status="up")` -> filter for `resource_type == "compute"`.

**Project allocation summary (two hop, auth):**
> User: "How much time is left on my ChemGraph allocation on Polaris?"
> 1. `alcf_account_list_projects()` -> find the ChemGraph project, grab its UUID.
> 2. `alcf_account_list_allocations(project_id=<uuid>)` -> filter by machine == Polaris.

**Recent job status (multi-hop, auth):**
> User: "Show me the status of my most recent Polaris job."
> 1. `alcf_compute_list_jobs(machine="polaris", historical=true, limit=1)`.
> 2. `alcf_compute_get_job_status(machine="polaris", job_id=<from step 1>, historical=true)` -- only if step 1's inline status is insufficient.

**Multi-tool health check (three hop, auth):**
> User: "For every compute machine that's up, tell me whether I have allocation left and how many jobs I have in the queue."
> 1. `alcf_status_list_resources(current_status="up", resource_type="compute")`.
> 2. `alcf_account_list_projects()` -> `alcf_account_list_allocations(...)` per project.
> 3. `alcf_compute_list_jobs(machine=<each>)` per active machine.

## Common failure modes

- **`IRI 401: Facility Specific authentication failed`** -- token missing/expired. The error text names the exact auth tools to call next.
- **`IRI 403`** -- the ALCF identity doesn't have access to that project or filesystem. Not something the agent can fix; report to user.
- **`No identity found in the session info`** -- Globus session lacks an ALCF identity linkage. See "Identity gotcha" in Auth section above.
- **`ALCF_IRI_ALLOW_UNSAFE=1 required`** -- write action attempted without the env-var gate. Report the exact requirement to the user; do not silently downgrade to a read.
- **Filesystem 404 / permission errors on `ls`** -- user's ALCF identity may not be provisioned in ALCF's ops table yet. Unrelated to this skill; direct them to ALCF support.

## References

- ALCF IRI docs: https://docs.alcf.anl.gov/services/iri-api/
- OpenAPI spec: https://api.alcf.anl.gov/openapi.json
- MCP server module: `chemgraph.mcp.alcf_iri_mcp`
  - `python -m chemgraph.mcp.alcf_iri_mcp` -- flat (43 tools, default)
  - `python -m chemgraph.mcp.alcf_iri_mcp --variant category` -- 7 dispatchers with discovery
  - env override: `CHEMGRAPH_IRI_MCP_VARIANT=flat|category`
- LangGraph workflow shipping the same tool set: `single_agent_iri` (see `examples/iri/`)
