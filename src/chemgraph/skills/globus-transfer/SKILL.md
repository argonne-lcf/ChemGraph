---
name: globus-transfer
description: Move files between Globus collections, such as a laptop running Globus Connect Personal and ALCF Eagle, Home, or Flare, with the official globus CLI. Use to stage inputs for HPC runs and collect large or binary outputs, including for the iri-hpc and pbs-hpc skills.
license: Apache-2.0
compatibility: Requires globus-cli and a Globus login by the user; on a laptop, Globus Connect Personal must be running.
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Globus transfers

Check authentication with `globus whoami`. For missing/expired authentication,
the user runs `globus login` in their terminal; for required collection consent,
they run the `globus login --gcs …` or consent command the CLI prints. Never run
interactive logins and never handle tokens. DNS, connection, or timeout errors
do not establish an authentication failure. A disconnected local endpoint needs
Globus Connect Personal running and reachable; repeating login does not fix that.
Report the actual error and use the [CLI diagnostics](references/cli.md#wait-and-check)
for failed or stalled tasks, preserving their records.

Collection IDs and the path each collection uses come from the selected configuration
(for `iri-hpc`, the workspace's `target.toml`) or explicit user input. Example
paths do not identify a workspace or authorize searching other configurations.
Verify that the local path maps the selected workspace, especially after importing
a configuration. Ask for missing values; never guess them. Collection
paths can differ from compute-node paths: on ALCF Eagle, collection
`/<project>/…` is `/eagle/<project>/…` on compute nodes. Use the exact commands in
[the CLI recipes](references/cli.md).

## Transfer

1. **Check** both sides with `globus ls`.
2. **Record** the transfer before submitting it. Records live in a state directory
   outside every transferred tree (for `iri-hpc`, `<workspace>/.globus/`). Write
   `STATE/<label>.json` with a new submission ID, the source, the destination, the
   label, and the options, as its own command. Never overwrite a record. Labels are
   unique per run and direction, for example `run-001-stage-in` or `run-001-collect`.
3. **Submit** from the record and save the returned task ID as `STATE/<label>.id`.
4. **Wait** with `globus task wait`, then check `globus task show`. The transfer is
   done only at `SUCCEEDED`.
5. **Confirm** the destination with `globus ls --all`.

## Rules

Use the Bash wrapper and fresh variable setup in every reference recipe. With
DeepAgent, allow `timeout=660` for Globus wait. After exit 124, inspect the saved
record/task and resume it; never generate a replacement submission ID.

- Resume from saved evidence. With `STATE/<label>.id`, wait on that task. With only
  the record, submit again from the same record: Globus returns the original task
  (`"code": "Duplicate"`) instead of starting a second one. If that resubmission
  fails, report it and retry the same record later.
- Never write a second record for a label. A replacement needs the user's approval
  after `globus task show` reports a terminal state for any known task, and the
  active-task listing in the recipes succeeds with an empty result. A failed listing
  leaves the outcome uncertain.
- Report a failed or paused task with `globus task event-list`.
- Never delete files at either end unless the user asks.

For tracked HPC runs:

- Stage only before job submission. Once the run directory has `submission.started`
  or `job.id` or `submission.history`, never transfer into its remote counterpart
  again, including after resolving a rejected submission.
- Collect into a subdirectory such as `RUN_DIR/outputs/`, never into the run
  directory itself, so a transfer cannot overwrite job markers or inputs.

For other operations, consult the [Globus CLI reference](https://docs.globus.org/cli/)
and [ALCF's Globus guide](https://docs.alcf.anl.gov/data-management/data-transfer/using-globus/).
