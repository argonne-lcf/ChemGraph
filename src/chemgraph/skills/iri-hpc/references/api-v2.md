# IRI API v2 recipes

Adapted from the [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/).
Run each whole block from the workspace holding `target.toml`; set `RUN` inside
it. Each block initializes its own variables. The literal fd 3 wrapper works under
DeepAgent's `/bin/sh` as well as Bash and keeps child stdin closed. Do not replace
it with a quoted `bash -c` string or feed the script to Bash on stdin.

Sourcing `iri.sh` loads IDs and paths from `target.toml`. Request bodies contain no
secrets; the token travels only through a pipe from builtin `printf` to `curl -H @-`.
With DeepAgent `execute`, use `timeout=180` for submission and `timeout=420` for
IRI filesystem polling. After exit 124, inspect saved state; never infer that a
mutation was not sent. For `iri_submit.py`, rerun the normal helper after inspection:
it checks the OS lock and saved outcome, refusing uncertain replays. On a busy
exit 3, wait before invoking it again. `.submission.lock` is permanent, so its
presence does not indicate activity. Never delete it or any submission evidence.

HTTP errors return nonzero with the problem body printed. Report it rather than
retrying a write blindly.

## Discover

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
curl -sS --fail-with-body "$BASE/status/resources/$COMPUTE" | jq '{name, current_status}'
iri "$BASE/account/whoami"
iri "$BASE/account/projects" | jq -r '.[].name'
BASH
```

## Stage

ALCF's IRI `upload` and `download` return 501. Create the directory here, transfer
the files with the `globus-transfer` skill, then list them:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
[ ! -e "$RUN/submission.started" ] && [ ! -e "$RUN/job.id" ] &&
  [ ! -e "$RUN/submission.history" ] || { echo "Submission evidence exists; never restage" >&2; exit 1; }
[ -d "$RUN" ] || { echo "No local run directory $RUN" >&2; exit 1; }
iri_fs mkdir "$(jq -nc --arg path "$REMOTE_RUN" '{path: $path, parent: true}')" >/dev/null || exit 1
echo "Stage with STATE=.globus LABEL=$RUN-stage-in" \
  "SOURCE=$LOCAL:$LOCAL_ROOT/$RUN/ DESTINATION=$REMOTE_COLLECTION:$REMOTE_GLOBUS_ROOT/$RUN/"
BASH
```

Only after that command succeeds, write the `globus-transfer` record with exactly
those values, transfer, and wait until `SUCCEEDED`. Then list the staged inputs:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
iri_fs ls "$(jq -nc --arg path "$REMOTE_RUN" '{path: $path}')" |
  jq -r '.result.output[] | "\(.type) \(.size) \(.name)"'
BASH
```

If `mkdir` reports that the path already exists, this run was staged before: resume
from its `.globus/$RUN-stage-in` record, or choose a new `RUN`; never stage into it
again.

## Submit

`request.json` wraps a PSI/J JobSpec. Use absolute remote paths and select
resources here:

```json
{
  "resource_id": "<compute_resource_id>",
  "job_spec": {
    "name": "cg-water-opt",
    "executable": "/bin/bash",
    "arguments": ["<REMOTE_RUN>/launch.sh"],
    "directory": "<REMOTE_RUN>",
    "stdout_path": "<REMOTE_RUN>/job.out",
    "stderr_path": "<REMOTE_RUN>/job.err",
    "resources": {"node_count": 1},
    "attributes": {
      "duration": 300,
      "queue_name": "debug",
      "account": "<account>",
      "custom_attributes": {"filesystems": "home:eagle"}
    }
  }
}
```

The helper omits `Idempotency-Key`, so submission does not require a server-side
idempotency store. Its local marker blocks uncertain replays. Submit separately
from staging, only after the transfer succeeded and inputs were checked:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
python "$SKILLS/iri-hpc/scripts/iri_submit.py" "$RUN"
BASH
```

Exit 0 prints only the full job ID; 1 means invalid input/definite rejection;
3 means uncertain, busy, or inconsistent evidence. If persistence fails after a
response, stderr includes the received outcome (and any job ID) for investigation;
stdout stays empty and the marker continues to block uncertain replays. `--retry-uncertain` always
refuses. A timeout, 409, redirect, missing response ID, or 5xx (including a new
headerless 501) stays blocked. No automatic fallback or second POST occurs.

## Monitor

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
JOB=$(cat "$RUN/job.id") && [ -n "$JOB" ] || exit 1
iri "$BASE/compute/status/$COMPUTE/$JOB?historical=true" |
  jq '{id, state: .status.state, exit_code: .status.exit_code, message: .status.message}'
BASH
```

States are new, queued, held, active, completed, failed, and canceled. If a facility
operator independently confirms an uncertain run's job ID, use this read-only
variant. It neither writes `job.id` nor clears the marker:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
JOB=...  # full job ID independently confirmed for this run
iri "$BASE/compute/status/$COMPUTE/$JOB?historical=true" |
  jq '{id, state: .status.state, exit_code: .status.exit_code, message: .status.message}'
BASH
```

## Inspect uncertainty

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
: "${OWNER:?target.toml must identify the user}"
# The user's jobs: an empty listing is diagnostic, never permission to resend.
offset=0
while :; do
  page=$(iri -X POST -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg owner "$OWNER" '{owner: $owner}')" \
    "$BASE/compute/status/$COMPUTE?historical=true&limit=100&offset=$offset") &&
    jq -e 'type == "array"' <<<"$page" >/dev/null || exit 1
  jq -c '.[] | {id, state: .status.state, account: .job_spec.attributes.account,
    queue: .job_spec.attributes.queue_name}' <<<"$page" || exit 1
  [ "$(jq length <<<"$page")" -lt 100 ] && { echo "all pages read"; break; }
  offset=$((offset + 100))
done
BASH
```

A failed/malformed page leaves the search incomplete. IRI listings lack job names
and paths. Neither a complete listing nor user approval makes a generic resend
safe. Preserve the uncertain marker and involve the facility operator.

## Resolve a legacy idempotency rejection

Only the old **headered** helper's HTTP 501 with problem status 501 and this exact
detail is eligible:

```text
Idempotency-Key provided but no idempotency store is configured (set IRI_IDEMPOTENCY_STORE).
```

That rejection occurs before scheduler submission. `IRI_IDEMPOTENCY_STORE` is a
server setting, not a laptop fix. Confirm the request is unchanged and there is no
accepted job. Inspect **all** earlier attempts/session transcripts: old retries
could overwrite a timeout with a later 501. A latest 501 alone is insufficient;
any unknown/accepted attempt means stop.

Write a nonsecret evidence note identifying every reviewed attempt and why each
was rejected before submission. Present it to the user and obtain confirmation
before running this local recovery command:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
python "$SKILLS/iri-hpc/scripts/iri_submit.py" "$RUN" \
  --resolve-idempotency-rejection evidence.txt --confirm-all-attempts-rejected
BASH
```

Recovery performs no authentication, API calls, staging, or submission. It archives
the original marker/error texts and the note in immutable `submission.history`,
then clears only that matching legacy marker. It preserves `request.json`, staged
files, and old error snapshots. Exit 0 has **empty stdout** and reports on stderr
that no job was submitted. Repeat recovery only with the same evidence if cleanup
was interrupted; a newer marker is never cleared. After successful recovery, use
the separate normal submission block above. **Never restage this run.**

## Cancel

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
JOB=$(cat "$RUN/job.id") && [ -n "$JOB" ] || exit 1
iri -X DELETE -o /dev/null -w '%{http_code}\n' "$BASE/compute/cancel/$COMPUTE/$JOB"  # 204
BASH
```

## Collect

`view` needs `size`; without it the content is empty. Both calls return the text in
`.result.output.content`.

Write to a temporary file and replace the saved copy only with valid JSON, so a
failed call never erases an earlier result:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1
RUN=...  # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
tmp=$(mktemp "$RUN/.result.json.XXXXXX") &&
  iri_fs view "$(jq -nc --arg path "$REMOTE_RUN/result.json" '{path: $path, size: 1048576, offset: 0}')" |
  jq -je '.result.output.content | strings' > "$tmp" && jq -e . "$tmp" >/dev/null &&
  mv "$tmp" "$RUN/result.json" || { rm -f "$tmp"; echo "Collection failed; kept the previous result.json" >&2; }
iri_fs tail "$(jq -nc --arg path "$REMOTE_RUN/job.err" '{path: $path, lines: 50}')" |
  jq -j '.result.output.content'
BASH
```

Save collected content in the local run directory before interpreting it. Use the
`globus-transfer` skill for large or binary outputs.
