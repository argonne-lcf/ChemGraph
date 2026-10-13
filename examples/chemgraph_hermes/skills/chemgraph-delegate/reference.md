# Handoff reference

Read this for errors, inspection, and recovery. The normal workflow is in
[SKILL.md](SKILL.md).

## Handoff record

Keep one active task per Discord conversation and requesting user. Store this
record in the task directory and retain its absolute path in conversation context:

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

Use null for IDs/status not yet known. Save the process handle when launching,
the ChemGraph session ID as soon as the early receipt appears, and the final
status/IDs when the process finishes. The record supports routing; it is not an
authorization boundary. Never select a session merely because it is the newest.

## Receipts and errors

Receipt version 1 has `session_id`, `status`, `assistant_response`, `interrupts`,
`error`, `error_code`, and `exit_code`. A running receipt has empty response and
interrupts, with null exit/error fields. Final status describes the saved state
when known; exit/error fields describe the CLI invocation. An export error can
coexist with completed status. Missing session IDs/statuses are null.

| Exit code | Meaning and next action |
| --- | --- |
| `0` | Agent turn completed. Check scientific outputs and pending jobs. |
| `3` | Input is pending. Relay clarifications; use a terminal for other interrupts. |
| `1` | Initialization, execution, export, or configuration-file loading failed. Inspect errors/logs. |
| `2` | Invalid invocation, answers, or incompatible resume. Correct the problem. |
| `130` | KeyboardInterrupt was handled. Inspect saved state and external jobs. |

| `error_code` | Next action |
| --- | --- |
| `invalid_answers` | Correct the JSON/file or empty answer values; preserve the user's answers. |
| `stale_answers` | Inspect current IDs and reconcile the replies; do not blindly resubmit. |
| `not_waiting` | Inspect the completed/failed session rather than submitting answers again. |
| `unsupported_interrupt` | Use interactive resume for tool reviews, unsupported payloads, or missing IDs. |
| `config` | Correct flags/configuration; an incompatible session may need its original environment. |
| `execution` | Inspect the saved state and logs; failed operations need an explicit interactive retry. |
| `export` | Inspect and export saved results without executing another query. |
| `interrupted` | Reconcile the stopped process and any external jobs before continuing. |

Error fields are null when no CLI error occurred. Unencodable interrupt payloads
become `unsupported` with a fixed inspection message; their IDs and exit **3**
survive. Never turn that placeholder into a clarification answer.

An unusable receipt destination returns exit **1** and diagnostics, but cannot
reliably contain its own error receipt. Argument parsing, colliding paths, or
termination before session creation can also leave no receipt. Always use a
fresh filename; never reuse a prior result when the current process produced none.

## Inspect and continue

First reconcile the known process handle and confirm it has stopped. Use Hermes's
advertised process tool (`process_manage`, or `process` on older versions) with
`action="poll"`/`action="log"` and the recorded process `session_id`.
Then inspect with the configured absolute executable:

```bash
/absolute/chemgraph-env/bin/chemgraph run --workflow main_agent \
  --dangerously-skip-approvals --resume SESSION_ID \
  --result-file /absolute/run-directory/inspection-001.json
```

This restores state without advancing the graph or calling model/tools. Do this
after a missed notification, gateway restart, interactive completion, export
failure, or uncertain submission. Compare pending IDs with the handoff record
before accepting or resubmitting replies. Serialize inspection and execution;
there is no cross-process session lock.

To re-export an existing result, use the same inspection command with
`--output last_message --output-file /absolute/run-directory/recovered.txt`.
Never add a replacement query just to recover a failed export.

For tool reviews, unsupported payloads, or missing IDs, present:

```bash
/absolute/chemgraph-env/bin/chemgraph run --interactive --workflow main_agent \
  --dangerously-skip-approvals --resume SESSION_ID
```

The user can answer in a terminal. Failed sessions require inspection and an
explicit `/retry`. `--resume ... --query ...` does not answer a pending question.
Keep the bypass flag when restoring these bypass sessions; reviewed sessions
must be restored without it, and cannot be upgraded through this handoff.

## Uncertain outcomes

SIGTERM, OOM, or an abrupt crash may leave only a running receipt. After the
process stops, that status means an uncertain outcome. The early ID identifies
the attempted session but does not guarantee that a checkpoint was saved.
Never automatically repeat an uncertain calculation or external submission.

If the ID was lost, inspect `chemgraph session list` and `session show ID`,
matching the task and artifact directory. If the match is ambiguous or the
checkpoint is missing, report uncertainty and request reconciliation.

Stopping the CLI does not cancel already submitted jobs. Preserve job IDs and
check/cancel them separately when requested. A completed agent turn does not
prove calculation convergence or completion of remote work. Preserve partial
results and report missing evidence without inventing values.
