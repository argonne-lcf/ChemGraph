# IRI API v2 recipes

Adapted from the [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/).
Each shell command starts fresh. Run it from the workspace and begin it with these
lines; set only `RUN`. Sourcing `iri.sh` loads `COMPUTE`, `STORAGE`, `OWNER`,
`ACCOUNT`, `REMOTE_ROOT`, `LOCAL`, `LOCAL_ROOT`, `REMOTE_COLLECTION`, and
`REMOTE_GLOBUS_ROOT` from `./target.toml`. Request bodies contain no secrets and may
be passed as arguments; the token travels only through a pipe from the builtin
`printf` into `curl -H @-`.

```bash
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh" || exit 1   # pipefail, BASE, helpers, IDs from target.toml
RUN=...                                   # plain run name, no slashes, e.g. water-opt-001
REMOTE_RUN="$REMOTE_ROOT/$RUN"
```

HTTP errors return nonzero with the problem body printed. Report it rather than
retrying a write blindly.

## Discover

```bash
curl -sS --fail-with-body "$BASE/status/resources/$COMPUTE" | jq '{name, current_status}'
iri "$BASE/account/whoami"
iri "$BASE/account/projects" | jq -r '.[].name'
```

## Stage

ALCF's IRI `upload` and `download` return 501. Create the directory here, transfer
the files with the `globus-transfer` skill, then list them:

```bash
[ -d "$RUN" ] || { echo "No local run directory $RUN" >&2; exit 1; }
iri_fs mkdir "$(jq -nc --arg path "$REMOTE_RUN" '{path: $path, parent: true}')" >/dev/null || exit 1
echo "Stage with STATE=.globus LABEL=$RUN-stage-in" \
  "SOURCE=$LOCAL:$LOCAL_ROOT/$RUN/ DESTINATION=$REMOTE_COLLECTION:$REMOTE_GLOBUS_ROOT/$RUN/"
```

Only after that command succeeds, write the `globus-transfer` record with exactly
those values, transfer, and wait until `SUCCEEDED`. Then list the staged inputs:

```bash
iri_fs ls "$(jq -nc --arg path "$REMOTE_RUN" '{path: $path}')" |
  jq -r '.result.output[] | "\(.type) \(.size) \(.name)"'
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

Submit with `python "$SKILLS/iri-hpc/scripts/iri_submit.py" "$RUN"`. Exit 0 prints the full
job ID; exit 3 means uncertain; exit 1 means rejected or invalid input.

## Monitor and inspect uncertainty

```bash
JOB=$(cat "$RUN/job.id")
iri "$BASE/compute/status/$COMPUTE/$JOB?historical=true" |
  jq '{id, state: .status.state, exit_code: .status.exit_code, message: .status.message}'

# The user's jobs, to check an uncertain submission. IRI returns no job names or
# paths, so compare account, queue, and state with the run. "No job" holds only
# if the loop prints "all pages read"; any failed or malformed page stops it.
offset=0
while page=$(iri -X POST -H 'Content-Type: application/json' \
    -d "$(jq -nc --arg owner "$OWNER" '{owner: $owner}')" \
    "$BASE/compute/status/$COMPUTE?historical=true&limit=100&offset=$offset") &&
  jq -e 'type == "array"' <<<"$page" >/dev/null &&
  jq -c '.[] | {id, state: .status.state, account: .job_spec.attributes.account,
    queue: .job_spec.attributes.queue_name}' <<<"$page"; do
  [ "$(jq length <<<"$page")" -lt 100 ] && { echo "all pages read"; break; }
  offset=$((offset + 100))
done
```

States are new, queued, held, active, completed, failed, and canceled. If a page
fails, the search is incomplete and the submission stays uncertain.

## Cancel

```bash
iri -X DELETE -o /dev/null -w '%{http_code}\n' "$BASE/compute/cancel/$COMPUTE/$JOB"  # 204
```

## Collect

`view` needs `size`; without it the content is empty. Both calls return the text in
`.result.output.content`.

Write to a temporary file and replace the saved copy only with valid JSON, so a
failed call never erases an earlier result:

```bash
tmp=$(mktemp "$RUN/.result.json.XXXXXX") &&
  iri_fs view "$(jq -nc --arg path "$REMOTE_RUN/result.json" '{path: $path, size: 1048576, offset: 0}')" |
  jq -je '.result.output.content | strings' > "$tmp" && jq -e . "$tmp" >/dev/null &&
  mv "$tmp" "$RUN/result.json" || { rm -f "$tmp"; echo "Collection failed; kept the previous result.json" >&2; }
iri_fs tail "$(jq -nc --arg path "$REMOTE_RUN/job.err" '{path: $path, lines: 50}')" |
  jq -j '.result.output.content'
```

Save collected content in the local run directory before interpreting it. Use the
`globus-transfer` skill for large or binary outputs.
