---
name: alcf_iri_bash
description: Query ALCF's IRI Facility API from a bash-capable coding agent using curl + jq. Use for any question about ALCF machines (Aurora, Crux, Polaris, Sophia), ALCF projects/allocations, PBS job state, or ALCF filesystem contents. Prefer this over guessing; the API is authoritative.
runtime: bash
prerequisites:
  - curl
  - jq
env_vars:
  - ALCF_API_TOKEN (required for authenticated endpoints; see Auth section)
  - ALCF_IRI_ALLOW_UNSAFE (required for write actions; default unset)
---

## When to use

Fire this skill any time the user asks about:

- **ALCF machine state** -- "Is Aurora up?", "Which machines are in maintenance?"
- **Incidents / outages** -- "Any recent Polaris outages?"
- **Projects and allocations** -- "What projects am I on?", "How much time is left on my ChemGraph allocation?"
- **PBS jobs** -- "Show my last completed job on Polaris", "What's the status of job 246115?"
- **Job submission / cancellation** -- requires `ALCF_IRI_ALLOW_UNSAFE=1`.
- **ALCF filesystem** -- ls / stat / read / write on `/eagle`, `/home`.

Do **not** use this for general HPC questions unrelated to ALCF, or for reading local files.

## Base URL and auth

```
BASE=https://api.alcf.anl.gov/api/v1
```

Public endpoints (all `facility.*`, all `status.*`, and `account.list_capabilities` / `get_capability`) need **no auth**. Everything else needs a bearer token.

**Getting a token, in order of preference:**

1. **Env var already set** -- `echo $ALCF_API_TOKEN` prints a JWT. Use it directly.
2. **On-disk Globus cache** -- `~/.globus/app/8b84fc2d-49e9-49ea-b54d-b3a29a70cf31/alcf_facility_api_app/tokens.json`. When this file exists, extract with:
    ```bash
    TOK=$(jq -r '.access_token // .data.DEFAULT."6be511f6-a071-471f-9bc0-02a0d0836723".access_token' \
        ~/.globus/app/8b84fc2d-49e9-49ea-b54d-b3a29a70cf31/alcf_facility_api_app/tokens.json)
    export ALCF_API_TOKEN=$TOK
    ```
    (The two paths cover both cache-file shapes: flat and nested.)
3. **ALCF's helper CLI** -- when neither of the above works, run:
    ```bash
    mkdir -p ~/tools
    curl -o ~/tools/alcf_facility_api_globus_token.py \
        https://raw.githubusercontent.com/argonne-lcf/alcf-facility-api-token/refs/heads/main/alcf_facility_api_globus_token.py
    python ~/tools/alcf_facility_api_globus_token.py authenticate
    export ALCF_API_TOKEN=$(python ~/tools/alcf_facility_api_globus_token.py get_access_token)
    ```
    `authenticate` opens an interactive Globus flow (URL to visit + code to paste). **Sign in with `<user>@alcf.anl.gov`, NOT `<user>@anl.gov`** -- only the `alcf.anl.gov` identity grants IRI access.

**All authenticated requests use:**
```bash
curl -s -H "Authorization: Bearer $ALCF_API_TOKEN" "$BASE/<path>"
```

**On 401** the response body will say `token expired` or `No identity found in the session info`. Refresh via option 3 above and retry. Do not retry more than twice.

## Safety gate for writes

Write actions (`POST /compute/job/...`, `DELETE /compute/cancel/...`, `POST /filesystem/mkdir`, `DELETE /filesystem/rm`, `PUT /filesystem/chmod`, etc.) **only work when the server operator has set `ALCF_IRI_ALLOW_UNSAFE=1`**. When targeting the ChemGraph MCP or LangGraph wrapper, this is enforced client-side and returns `RuntimeError: Action X modifies HPC state...`. When calling the raw API via curl, the API itself may not gate you -- **do not submit write requests unless the user explicitly asked for that write**. Report the operation you're about to perform before running it.

## Resource-name resolution

Endpoints under `/compute/`, `/filesystem/`, and any `/status/resources/{id}` take a **UUID**, not a name. When the user says "aurora" or "crux", resolve first:

```bash
# Look up a compute machine UUID by name (case-insensitive on the API side; jq matches literal)
UUID=$(curl -s "$BASE/status/resources" | jq -r '.[] | select(.name | ascii_downcase == "aurora") | .id')
```

For **filesystem** endpoints, the UUID must be a **storage** resource (`eagle`, `home`), not a compute resource. Pick by path prefix: `/eagle/...` -> `eagle`, `/home/...` -> `home`. Passing a compute UUID to `/filesystem/*` returns 400. `/flare/...` is not exposed by IRI at all -- tell the user to use `scp` for `/flare` paths.

## Endpoint recipes

Each recipe shows the raw curl call plus the `jq` filter to extract the specific answer. All GET unless noted. All auth-required unless marked `[public]`.

### Facility metadata `[public]`

```bash
curl -s "$BASE/facility"                                        # name, sites
curl -s "$BASE/facility/sites"                                  # list all sites
curl -s "$BASE/facility/sites/$SITE_UUID"                       # one site's detail
```

### Resource status `[public]`

```bash
curl -s "$BASE/status/resources"                                # all resources (~10 entries)
curl -s "$BASE/status/resources?resource_type=compute&current_status=up"
curl -s "$BASE/status/resources/$UUID"                          # one resource
curl -s "$BASE/status/incidents"                                # outages + scheduled maintenance
curl -s "$BASE/status/events"                                   # state-change history
```

Common `jq` follow-ups:
```bash
# How many total resources?
curl -s "$BASE/status/resources" | jq 'length'

# How many compute resources currently up?
curl -s "$BASE/status/resources" | jq '[.[] | select(.resource_type=="compute" and .current_status=="up")] | length'

# UUID of a resource by name (case-insensitive)
curl -s "$BASE/status/resources" | jq -r '.[] | select(.name | ascii_downcase == "aurora") | .id'
```

### Capabilities `[public]`

```bash
curl -s "$BASE/account/capabilities"                            # list all
curl -s "$BASE/account/capabilities/$CAP_UUID"                  # one capability

# ID of a capability by name
curl -s "$BASE/account/capabilities" | jq -r '.[] | select(.name=="aurora") | .id'
```

### Account, projects, allocations `[auth]`

```bash
H=(-H "Authorization: Bearer $ALCF_API_TOKEN")

curl -s "${H[@]}" "$BASE/account/projects"                                                # your projects
curl -s "${H[@]}" "$BASE/account/projects/$PROJECT_UUID"                                  # one project
curl -s "${H[@]}" "$BASE/account/projects/$PROJECT_UUID/project_allocations"              # all allocations for a project
curl -s "${H[@]}" "$BASE/account/projects/$PROJECT_UUID/project_allocations/$ALLOC_UUID"  # one allocation detail
curl -s "${H[@]}" "$BASE/account/projects/$PROJECT_UUID/project_allocations/$ALLOC_UUID/user_allocations"  # per-user slice
```

Common jq:
```bash
# Find a project by name and cache its UUID
PID=$(curl -s "${H[@]}" "$BASE/account/projects" | jq -r '.[] | select(.name=="ChemGraph") | .id')

# Count user_ids on that project
curl -s "${H[@]}" "$BASE/account/projects/$PID" | jq '.user_ids | length'

# Total allocations across all machines for the project
curl -s "${H[@]}" "$BASE/account/projects/$PID/project_allocations" | jq 'length'

# Sum node_hours across the project's allocations
curl -s "${H[@]}" "$BASE/account/projects/$PID/project_allocations" | jq '[.[].node_hours_allocated] | add'

# Highest usage:allocation ratio, return capability name
curl -s "${H[@]}" "$BASE/account/projects/$PID/project_allocations" \
    | jq -r 'max_by(.node_hours_used / .node_hours_allocated) | .capability_name'
```

### Compute (PBS jobs) `[auth]`

```bash
H=(-H "Authorization: Bearer $ALCF_API_TOKEN")

# Status of one job (add historical=true if the job is completed)
curl -s "${H[@]}" "$BASE/compute/status/$MACHINE_UUID/$JOB_ID"
curl -s "${H[@]}" "$BASE/compute/status/$MACHINE_UUID/$JOB_ID?historical=true"

# List jobs -- NOTE this is POST in the API even though it is read-only.
# Active queue (default):
curl -s -X POST "${H[@]}" "$BASE/compute/status/$MACHINE_UUID"
# Completed / historical:
curl -s -X POST "${H[@]}" "$BASE/compute/status/$MACHINE_UUID?historical=true&limit=100&offset=0"

# Submit a job (WRITE, gated behind ALCF_IRI_ALLOW_UNSAFE=1 in wrappers)
curl -s -X POST "${H[@]}" -H 'Content-Type: application/json' \
    "$BASE/compute/job/$MACHINE_UUID" \
    -d '{"executable":"/bin/bash",
         "arguments":["-lc","echo hi; sleep 10"],
         "name":"my_job",
         "stdout_path":"/home/<user>/out",
         "stderr_path":"/home/<user>/err",
         "resources":{"node_count":1},
         "attributes":{"duration":300,"queue_name":"debug","account":"<project>",
                       "custom_attributes":{"filesystems":"home:eagle"}}}'
# NOTE `filesystems` is COLON-SEPARATED STRING, not a list.

# Cancel a job (WRITE)
curl -s -X DELETE "${H[@]}" "$BASE/compute/cancel/$MACHINE_UUID/$JOB_ID"
```

Common jq for job questions:
```bash
POLARIS=$(curl -s "$BASE/status/resources" | jq -r '.[] | select(.name | ascii_downcase == "polaris") | .id')

# Total jobs currently in Polaris active queue
curl -s -X POST "${H[@]}" "$BASE/compute/status/$POLARIS" | jq 'length'

# Count in a specific state
curl -s -X POST "${H[@]}" "$BASE/compute/status/$POLARIS" | jq '[.[] | select(.status.state=="queued")] | length'

# Oldest queued job's numeric PBS id (strip the .polaris-pbs-01.hsn... suffix)
curl -s -X POST "${H[@]}" "$BASE/compute/status/$POLARIS" \
    | jq -r '[.[] | select(.status.state=="queued")] | min_by(.job_spec.attributes.submit_time // .id) | .id | split(".")[0]'
```

**Pagination gotcha:** the list-jobs endpoint pages at 100. When counting "all jobs in the queue" and the response has 100 entries, page again with `offset=100`, `offset=200`, ... until fewer than `limit` come back. **Don't report a count without paginating -- 100 in the response usually means "at least 100."**

### Filesystem `[auth]`

**Reminder:** `machine` here is a **storage** UUID (`eagle`, `home`), not compute.

```bash
H=(-H "Authorization: Bearer $ALCF_API_TOKEN")
EAGLE=$(curl -s "$BASE/status/resources" | jq -r '.[] | select(.name | ascii_downcase == "eagle") | .id')

# ls a directory
curl -s "${H[@]}" "$BASE/filesystem/ls/$EAGLE?path=/eagle/<project>&showHidden=false&recursive=false"

# cat a file (small)
curl -s "${H[@]}" "$BASE/filesystem/cat/$EAGLE?path=/eagle/<project>/out.log"

# Head / tail
curl -s "${H[@]}" "$BASE/filesystem/head/$EAGLE?path=/eagle/<project>/out.log&lines=50"
curl -s "${H[@]}" "$BASE/filesystem/tail/$EAGLE?path=/eagle/<project>/out.log&lines=50"

# mkdir / rm / chmod (WRITE, gated)
curl -s -X POST "${H[@]}" "$BASE/filesystem/mkdir/$EAGLE?path=/eagle/<project>/newdir"
curl -s -X DELETE "${H[@]}" "$BASE/filesystem/rm/$EAGLE?path=/eagle/<project>/oldfile"
```

### Task queue `[auth]`

The `/task/*` endpoints track handles for long-running async ops that other endpoints may return.

```bash
H=(-H "Authorization: Bearer $ALCF_API_TOKEN")

curl -s "${H[@]}" "$BASE/task"                    # list tasks tied to your token
curl -s "${H[@]}" "$BASE/task/$TASK_UUID"         # one task's state
curl -s -X DELETE "${H[@]}" "$BASE/task/$TASK_UUID"   # cancel a task
```

## Multi-hop recipes for common asks

**"How many jobs do I have queued on Polaris?"**
```bash
POLARIS=$(curl -s "$BASE/status/resources" | jq -r '.[] | select(.name | ascii_downcase == "polaris") | .id')
curl -s -X POST -H "Authorization: Bearer $ALCF_API_TOKEN" "$BASE/compute/status/$POLARIS" \
    | jq '[.[] | select(.status.state=="queued")] | length'
```

**"What's the status of my most recent Polaris job?"**
```bash
H=(-H "Authorization: Bearer $ALCF_API_TOKEN")
POLARIS=$(curl -s "$BASE/status/resources" | jq -r '.[] | select(.name | ascii_downcase == "polaris") | .id')
# Newest historical job (limit=1)
JOB=$(curl -s -X POST "${H[@]}" "$BASE/compute/status/$POLARIS?historical=true&limit=1" | jq -r '.[0].id')
curl -s "${H[@]}" "$BASE/compute/status/$POLARIS/$JOB?historical=true"
```

**"For every machine that's up, tell me if my ChemGraph project has allocation left."**
```bash
H=(-H "Authorization: Bearer $ALCF_API_TOKEN")
UP=$(curl -s "$BASE/status/resources?resource_type=compute&current_status=up" | jq -r '.[].name')
PID=$(curl -s "${H[@]}" "$BASE/account/projects" | jq -r '.[] | select(.name=="ChemGraph") | .id')
ALLOCS=$(curl -s "${H[@]}" "$BASE/account/projects/$PID/project_allocations")
for m in $UP; do
    echo "== $m =="
    echo "$ALLOCS" | jq --arg m "$m" '.[] | select(.capability_name | ascii_downcase == ($m | ascii_downcase)) | {allocated: .node_hours_allocated, used: .node_hours_used}'
done
```

## Rules for the coding agent

1. **Don't invent UUIDs, names, or counts.** If a value isn't in a `curl` response you already ran, run another `curl` to get it.
2. **Paginate list_jobs whenever you see exactly the limit's worth of entries** -- there may be more.
3. **Never submit a write action the user didn't ask for.** Read-only work needs no confirmation; writes require explicit user consent and `ALCF_IRI_ALLOW_UNSAFE=1` (or the user calling the raw API themselves).
4. **On 401, refresh once** via the ALCF helper CLI (option 3 in Auth), then retry once. If it 401s again, report the error verbatim and stop -- do not loop.
5. **Report exact numbers, not rounded ones.** If `jq 'length'` says 43, say 43, not "about 40."
6. **When the user asks a scalar question, answer with that scalar.** "How many resources?" -> "10", not "there are 10 resources spanning compute, storage, and services." The rubric rewards direct answers.

## Common failure modes

- **`401 unauthorized`** -- token missing/expired/not-ALCF-identity. Follow Auth section.
- **`400` from `/filesystem/*`** -- you passed a compute UUID (aurora/crux/polaris) instead of a storage one (eagle/home).
- **`404 not found` on `/compute/status/.../{job_id}`** -- add `?historical=true`, the job may be completed.
- **Empty `[]` from list endpoints** -- your token is valid but you don't have jobs / projects / allocations in that scope. Report the empty result honestly, don't hallucinate.
- **`/filesystem/*` returning 500 with "identity mapping not configured"** -- an ALCF-side ops issue; not fixable from the agent side. Tell the user.

## References

- ALCF IRI docs: https://docs.alcf.anl.gov/services/iri-api/
- OpenAPI spec: https://api.alcf.anl.gov/openapi.json  (fetch and grep for endpoint shapes if a recipe here is stale)
- Companion MCP-oriented skill: `alcf_iri.md` (same capabilities, exposed via `chemgraph.mcp.alcf_iri_mcp` instead of raw curl)
