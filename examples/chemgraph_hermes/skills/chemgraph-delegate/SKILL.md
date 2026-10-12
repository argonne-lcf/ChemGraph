---
name: chemgraph-delegate
description: Delegate chemistry tasks to ChemGraph through its CLI from Hermes, monitor results, relay clarification questions and answers through Discord, and continue saved sessions. Use when ChemGraph should own the scientific workflow.
license: Apache-2.0
---

# Delegate to ChemGraph

Use Hermes terminal/process tools to run ChemGraph in its own Python environment.
Keep scientific planning and calculations inside ChemGraph. This skill uses a
trusted personal setup with headless approval bypass; it does not install a
service or grant permission for work beyond the user's request.

## Prepare and launch

1. Resolve the configured absolute ChemGraph executable, optional TOML config,
   and parent directory for runs. Check `chemgraph run --help` using that
   executable if its capabilities are unknown; require `--result-file` and
   `--answers-file`. Use existing model credentials;
   do not put credentials in queries. If setup is missing, report what is needed.
2. Build a self-contained task including inputs, calculator/model, constraints,
   and desired outputs. Preserve requested scientific settings; have ChemGraph
   ask about missing choices. Do not silently substitute a cheaper calculator.
3. Create a unique directory on the terminal host for this scientific task.
   Stage inputs there or supply absolute paths accessible to that host. A path
   on the user's laptop is not automatically visible on a remote terminal.
4. Set `CHEMGRAPH_LOG_DIR` to that directory for the child process and use it as
   `--workspace` and the Hermes terminal `workdir`. Quote query and path values
   for the host shell; never interpolate raw user text into shell syntax.
5. Keep one active task per Discord conversation (channel/thread) and requesting
   user. Record those IDs in the task's `handoff.json` before launching and keep
   its absolute path in the Hermes conversation context. If that pair already
   has a running or waiting task, continue it before starting another.

Example shell command after creating the directory:

```bash
CHEMGRAPH_LOG_DIR=/absolute/run-directory \
/absolute/chemgraph-env/bin/chemgraph run \
  --workflow main_agent --workspace /absolute/run-directory \
  --dangerously-skip-approvals --human-supervised \
  --query "SELF_CONTAINED_CHEMISTRY_TASK" \
  --result-file /absolute/run-directory/result-001.json
```

Add `--config /absolute/config.toml` when configured; CLI runs do not implicitly
load that file. Headless `main_agent` rejects `--mcp-url`, `--mcp-command`, and
their TOML equivalents. Report that limitation instead of changing the workflow.

Pass the launch command to `terminal` with `background=true`, `notify=true`, and
the absolute `workdir`; a headless run needs no PTY. For older Hermes versions,
use their advertised completion-notification option or monitor with the process tool.
Do not silently remove the bypass flag: headless `main_agent` requires it.
If default tool reviews are desired, use an interactive session instead.

## Monitor and report

- Save the returned Hermes process `session_id` in `handoff.json`. Use the available
  process tool (`process_manage`, or `process` on older versions) with
  `action="poll"` or `action="log"` and that `session_id` to
  inspect that process; avoid tight polling. Never launch it again merely because
  the original terminal call returned a running process.
- Read the fresh receipt with file tools and record its **ChemGraph session ID**,
  status, and pending IDs. It is different from the Hermes process handle. Never
  select a session merely because it is the most recent. A handoff record contains:

```json
{
  "conversation_id": "DISCORD_CHANNEL_OR_THREAD_ID",
  "requesting_user_id": "DISCORD_USER_ID",
  "workspace": "/absolute/run-directory",
  "chemgraph_session_id": "CHEMGRAPH_SESSION_ID",
  "process_session_id": "HERMES_PROCESS_HANDLE",
  "result_file": "/absolute/run-directory/result-001.json",
  "status": "waiting_for_user",
  "pending_interrupt_ids": ["INTERRUPT_ID"]
}
```

Use null for IDs/status not yet known. Update the record on launch and completion;
do not reuse a prior receipt if the new process fails to produce one. Read the
receipt's `assistant_response` and relevant output JSON/structures to substantiate
the answer; retain units, calculator/model, convergence, and artifact paths.

| Exit code | Meaning and next action |
| --- | --- |
| `0` | Agent turn completed. Inspect scientific results and any pending jobs. |
| `3` | Input is pending. Relay clarifications; use a terminal for other interrupts. |
| `1` | Execution, initialization, or output export failed. Inspect logs and saved state before retrying. |
| `2` | Invalid configuration or incompatible resume. Correct configuration; do not replace the task automatically. |
| `130` | Interrupted. Inspect saved state and any external jobs before continuing. |

A receipt can contain partial or empty assistant text on a paused/failed turn;
its existence is not success. Check `error` as well as status: an export can fail
after the session completes. Report tool errors and missing results without inventing
values. Completion of the agent turn is separate from calculation convergence
and completion of any submitted remote work.

## Continue or recover

- For `waiting_for_user`, relay each `clarification` payload's `question` to the
  original conversation and requesting user. Associate each reply with its exact
  pending ID. Accept replies only from that user in that conversation. Gather all
  answers; resolve ambiguity rather than guessing. Preserve the user's answers.
- Write a fresh UTF-8 JSON file mapping **all** pending IDs to nonempty strings,
  for example `{"INTERRUPT_ID": "Use EMT."}`. Resume with `run --workflow main_agent
  --dangerously-skip-approvals --resume SESSION_ID --answers-file /absolute/run-directory/answers-001.json
  --result-file /absolute/run-directory/result-002.json`. Launch and monitor it as
  above, updating the process handle. Exit `3` can introduce a new set of questions.
  Do not combine answers with `--query` or invent an approval for a tool review.
- For a completed session, launch the same executable with `run --workflow
  main_agent --dangerously-skip-approvals --resume SESSION_ID --query "FOLLOW_UP"`.
  Keep all launches serialized per ChemGraph session and use a fresh `--result-file`;
  the saved workspace and configuration take precedence on resume. Release the
  conversation's active-task association when finished, preserving the record.
- For `tool_review`, `unsupported`, or missing interrupt IDs, present
  `chemgraph run --interactive --workflow main_agent
  --dangerously-skip-approvals --resume SESSION_ID` using the configured executable.
  The user can answer in a terminal.
  `--resume ... --query ...` does not answer a pending question.
- After restart, an uncertain submission, interactive completion, or an export
  failure, inspect without rerunning tools: `run --workflow main_agent
  --dangerously-skip-approvals --resume SESSION_ID --result-file /absolute/run-directory/inspection-001.json`.
  First reconcile the known process handle so it cannot overlap a continuation.
  Compare the current pending IDs with the stored ones before accepting or
  resubmitting a reply. Stale answers exit `2`; inspect instead of blindly retrying.
- A failed session needs inspection and interactive `/retry`; do not issue an
  automatic replacement query. If the session ID was lost, inspect `chemgraph
  session list` and `session show ID`, matching the task and artifact directory.
  If no matching receipt is found, report uncertainty before rerunning work.
- On cancellation, stop the known Hermes process only when requested. Already
  submitted remote jobs require their own cancellation/status checks. Checkpoints
  do not keep unfinished processes alive across crashes or guarantee replay of
  interrupted external operations. Preserve scheduler/job IDs for reconciliation.
