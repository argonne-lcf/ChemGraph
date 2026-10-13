---
name: chemgraph-connect
description: Delegate chemistry research and local scientific tasks to ChemGraph, track progress, answer clarification questions, and recover reports across sessions.
---

# ChemGraph delegation

Use the `chemgraph_connect` toolset when the user wants ChemGraph to perform
chemistry research, inspect molecular structures, or use its scientific tools.
ChemGraph is a separate persistent service. Hermes coordinates the conversation;
ChemGraph owns the scientific reasoning and validation.

## Submit and track

1. Call `cg_submit` with a self-contained text description of the requested work,
   known inputs, constraints and desired output. Only text input is supported;
   local Hermes paths are not uploaded or automatically available to ChemGraph.
2. Preserve the returned `message_id`, `task.id` and `task.contextId`. A receipt
   with `acceptance=accepted` confirms durable acceptance, not completion.
3. Call `cg_status` using `task_id` to retrieve progress or the report. If it is
   still SUBMITTED or WORKING, tell the user the task ID and that work continues.
   Avoid repeated tight polling. This CLI setup has no automatic completion
   notifications; check again on the next user request.
4. Report scientific conclusions only from the returned report. COMPLETED means
   the graph/reporting turn finished; it does not independently establish that
   results are scientifically validated. Inspect validation metadata and caveats.

## Clarification and continuation

- For INPUT_REQUIRED, inspect `metadata.waiting_reason` and `metadata.interrupts`.
- For `clarification`, relay the questions to the user without inventing answers.
  Reply with `cg_submit`, the pending `task_id`, and the answer text. For one
  question include its `interrupt_id`. For multiple questions include
  `interrupt_responses`, an object mapping **every** current interrupt ID to its
  answer, plus a short nonempty `text` such as "Answers to the pending questions".
- For `approval`, `configuration` or `reconciliation`, direct the user to the
  Connect operator CLI. Natural-language permission cannot authorize an operation.
  Do not retrieve operator credentials or attempt approvals with terminal tools.
- After a task completes, submit follow-up work with its `context_id` and omit
  `task_id`. This creates a new task in the same scientific conversation. Start
  unrelated work without either ID. Only one task can be active per context.

## Recover safely

- On `acceptance=unknown`, the task may already have been accepted. Do not submit
  the work again with a new message ID. Use `cg_tasks` to locate it and `cg_status`
  to inspect the candidate's history, including its original `messageId`.
- If replay is needed, use the returned `message_id` and exactly the original
  text, context/task IDs and interrupt fields, including originally omitted IDs.
  Reusing the ID with changed content is rejected. There is no automatic retry.
- `acceptance=rejected` means this request was rejected; read its error before
  correcting it. `acceptance=not_sent` means local configuration is missing.
- Use `cg_tasks` after a Hermes restart or lost conversation. Optionally filter
  by `context_id`. Pass `nextPageToken` back as `page_token` with the same filter
  to read another page. Inspect history before associating a task with a request.
- An uncertain **scientific execution or external submission** requires operator
  reconciliation, even when the A2A message itself was delivered successfully.
  Never start a replacement task to bypass that state.

## Results and cancellation

- `cg_status` returns the text report; `cg_artifacts` lists authenticated artifact
  references. References do not mean files have been downloaded to Hermes.
- For a file, the user can run `chemgraph-connect artifact ARTIFACT_ID OUTPUT`
  with their peer credential (or `--token-env CHEMGRAPH_CONNECT_TOKEN`) and
  `--url "$CHEMGRAPH_CONNECT_URL"` to match the configured service origin. Do not
  expose credentials or pass authenticated URLs to unrelated services.
- Use `cg_cancel` only when the user wants to stop a task. Report cancellation as
  confirmed only when the service returns CANCELED. If stopping is unconfirmed,
  keep tracking the existing task.

The initial local setup has no HPC endpoint, messaging delivery, uploaded files,
or downloaded MLIP models. Do not claim those capabilities are configured.
