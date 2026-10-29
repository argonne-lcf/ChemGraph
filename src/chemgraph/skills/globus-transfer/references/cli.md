# Globus CLI recipes

Adapted from the [Globus CLI reference](https://docs.globus.org/cli/reference/).
Run each whole block from the workspace. Set its variables inside the literal fd 3
Bash wrapper; each execution starts fresh. Bash reads the script from fd 3 while
child stdin stays closed, including under DeepAgent's `/bin/sh`. Do not substitute
a quoted `bash -c` or a stdin heredoc. Bash and `/dev/fd` support are required.

## Find collections and paths

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
LOCAL=...  # configured local collection ID
EAGLE=...  # configured Eagle collection ID
globus endpoint local-id                       # this laptop's Globus Connect Personal
globus endpoint search "alcf#dtn_eagle"        # ALCF Eagle mapped collection
globus ls "$LOCAL:/~/"                         # Globus Connect Personal home
globus ls "$EAGLE:/<project>/"                 # Eagle: compute path /eagle/<project>/
BASH
```

## Record and submit a transfer

For IRI staging, use the exact SOURCE and DESTINATION values printed by the
iri-hpc Stage recipe. For other transfers, use the selected collections and paths.

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
STATE=...        # state directory outside the transferred tree, e.g. <workspace>/.globus
LABEL=...        # unique per run and direction, e.g. run-001-stage-in
SOURCE=...       # configured source collection:path
DESTINATION=...  # configured destination collection:path
mkdir -p "$STATE"
set -C       # refuse to overwrite an existing record
sid=$(globus task generate-submission-id) && [ -n "$sid" ] &&
jq -n --arg sid "$sid" \
  --arg source "$SOURCE" --arg destination "$DESTINATION" \
  --arg lbl "$LABEL" \
  '{submission_id: $sid, source: $source, destination: $destination,
    label: $lbl, options: ["--recursive"]}' > "$STATE/$LABEL.json"
BASH
```

Submit from the record, as a separate command. Submitting again from the same
record is safe: Globus answers `"code": "Duplicate"` with the original task ID.

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
STATE=...
LABEL=...
R="$STATE/$LABEL.json"
jq -e '.options | type == "array" and all(type == "string")' "$R" >/dev/null || exit 1
options=()
while IFS= read -r option; do options+=("$option"); done < <(jq -r '.options[]' "$R")
task=$(globus transfer "$(jq -r .source "$R")" "$(jq -r .destination "$R")" \
  "${options[@]}" --label "$(jq -r .label "$R")" \
  --submission-id "$(jq -r .submission_id "$R")" --jmespath task_id --format unix) &&
  printf '%s\n' "$task" > "$STATE/$LABEL.id"
BASH
```

## Wait and check

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
STATE=...
LABEL=...
TASK=$(cat "$STATE/$LABEL.id")
globus task wait "$TASK" --timeout 600 --polling-interval 15 --timeout-exit-code 50
globus task show "$TASK" --jmespath '{status: status, files: files, faults: faults}' --format json
globus task event-list "$TASK" --filter-errors --limit 5   # when FAILED or stalled
globus ls --all "$(jq -r .destination "$STATE/$LABEL.json")"
BASH
```

`task wait` exits 0 on success, 1 on failure, and 50 on timeout. On timeout, keep
waiting on the same task. For DeepAgent `execute`, use `timeout=660` here. Exit
124 also requires inspecting/resuming the recorded task, not creating a new record.

## Before replacing a record

Run this only when the user approves a replacement and no known task is active:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
LABEL=...
globus task list --exact --filter-label "$LABEL" \
  --filter-status ACTIVE --filter-status INACTIVE --format json --jmespath 'DATA'
BASH
```

Continue only if the command succeeds and prints `[]`.
