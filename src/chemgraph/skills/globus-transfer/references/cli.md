# Globus CLI recipes

Adapted from the [Globus CLI reference](https://docs.globus.org/cli/reference/).
Each shell command starts fresh, so set the variables at the start of every command.

## Find collections and paths

```bash
globus endpoint local-id                       # this laptop's Globus Connect Personal
globus endpoint search "alcf#dtn_eagle"        # ALCF Eagle mapped collection
globus ls "$LOCAL:/~/"                         # Globus Connect Personal home
globus ls "$EAGLE:/<project>/"                 # Eagle: compute path /eagle/<project>/
```

## Record and submit a transfer

```bash
STATE=...    # state directory outside the transferred tree, e.g. <workspace>/.globus
LABEL=...    # unique per run and direction, e.g. run-001-stage-in
mkdir -p "$STATE"
set -C       # refuse to overwrite an existing record
sid=$(globus task generate-submission-id) && [ -n "$sid" ] &&
jq -n --arg sid "$sid" \
  --arg source "$LOCAL:/~/iri-test/run-001/" \
  --arg destination "$EAGLE:/<project>/<user>/iri-runs/run-001/" \
  --arg label "$LABEL" \
  '{submission_id: $sid, source: $source, destination: $destination,
    label: $label, options: ["--recursive"]}' > "$STATE/$LABEL.json"
```

Submit from the record, as a separate command. Submitting again from the same
record is safe: Globus answers `"code": "Duplicate"` with the original task ID.

```bash
R="$STATE/$LABEL.json"
jq -e '.options | type == "array" and all(type == "string")' "$R" >/dev/null || exit 1
options=()
while IFS= read -r option; do options+=("$option"); done < <(jq -r '.options[]' "$R")
task=$(globus transfer "$(jq -r .source "$R")" "$(jq -r .destination "$R")" \
  "${options[@]}" --label "$(jq -r .label "$R")" \
  --submission-id "$(jq -r .submission_id "$R")" --jmespath task_id --format unix) &&
  printf '%s\n' "$task" > "$STATE/$LABEL.id"
```

## Wait and check

```bash
TASK=$(cat "$STATE/$LABEL.id")
globus task wait "$TASK" --timeout 600 --polling-interval 15 --timeout-exit-code 50
globus task show "$TASK" --jmespath '{status: status, files: files, faults: faults}' --format json
globus task event-list "$TASK" --filter-errors --limit 5   # when FAILED or stalled
globus ls --all "$(jq -r .destination "$STATE/$LABEL.json")"
```

`task wait` exits 0 on success, 1 on failure, and 50 on timeout. On timeout, keep
waiting on the same task.

## Before replacing a record

Run this only when the user approves a replacement and no known task is active:

```bash
globus task list --exact --filter-label "$LABEL" \
  --filter-status ACTIVE --filter-status INACTIVE --format json --jmespath 'DATA'
```

Continue only if the command succeeds and prints `[]`.
