# IRI API v2 recipes

Adapted from the [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/).
Each shell command starts fresh. Begin every command by setting the variables
from the target configuration and sourcing the bundled helpers. Request bodies
contain no secrets and may be passed as arguments; the token travels only through
a pipe from the builtin `printf` into `curl -H @-`.

```bash
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
source "$SKILLS/iri-hpc/scripts/iri.sh"   # sets pipefail, BASE, iri, iri_fs, iri_task
COMPUTE=...      # compute_resource_id
STORAGE=...      # storage_resource_id matching the remote path prefix
REMOTE_RUN=...   # fresh remote run directory below remote_root
OWNER=...        # username: the ALCF username, not the whoami UUID
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
iri_fs mkdir "$(jq -nc --arg path "$REMOTE_RUN" '{path: $path, parent: true}')"
# ... globus-transfer stage-in of RUN_DIR, waited until SUCCEEDED ...
iri_fs ls "$(jq -nc --arg path "$REMOTE_RUN" '{path: $path}')" |
  jq -r '.result.output[] | "\(.type) \(.size) \(.name)"'
```

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

Submit with `python "$SKILLS/iri-hpc/scripts/iri_submit.py" RUN_DIR`. Exit 0 prints the full
job ID; exit 3 means uncertain; exit 1 means rejected or invalid input.

## Monitor and inspect uncertainty

```bash
JOB=$(cat RUN_DIR/job.id)
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
tmp=$(mktemp RUN_DIR/.result.json.XXXXXX) &&
  iri_fs view "$(jq -nc --arg path "$REMOTE_RUN/result.json" '{path: $path, size: 1048576, offset: 0}')" |
  jq -je '.result.output.content | strings' > "$tmp" && jq -e . "$tmp" >/dev/null &&
  mv "$tmp" RUN_DIR/result.json || { rm -f "$tmp"; echo "Collection failed; kept the previous result.json" >&2; }
iri_fs tail "$(jq -nc --arg path "$REMOTE_RUN/job.err" '{path: $path, lines: 50}')" |
  jq -j '.result.output.content'
```

Save collected content in the local run directory before interpreting it. Use the
`globus-transfer` skill for large or binary outputs.
