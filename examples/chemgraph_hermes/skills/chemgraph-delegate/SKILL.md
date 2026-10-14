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

Require a gateway user allowlist and a dedicated unprivileged OS user or
container. Approval bypass permits host shell access; the workspace is not a
sandbox. The handoff record and model instructions do not enforce identity.

## Launch

1. Resolve the absolute ChemGraph executable, optional TOML config, and run parent.
   Require `--result-file` and `--answers-file` in `run --help`. Use configured
   credentials without including them in queries. Report missing setup.
2. Preserve the user's scientific inputs, method, constraints, and requested
   outputs. Ask ChemGraph to clarify missing choices; do not substitute a method.
3. Create a unique run directory on the terminal host and stage accessible inputs.
   Keep one active task per Discord conversation and requesting user. Record that
   association, workspace, process handle, session ID, receipt path, status, and
   pending IDs in `handoff.json`; retain its path in the conversation context.
   Use null for unknown values. See [reference.md](reference.md) for its shape.
4. Launch with the directory as `CHEMGRAPH_LOG_DIR`, `--workspace`, and terminal
   `workdir`. Quote paths and queries safely; never interpolate raw user text into
   shell syntax. Use a fresh receipt path on every invocation.

```bash
CHEMGRAPH_LOG_DIR=/absolute/run-directory \
/absolute/chemgraph-env/bin/chemgraph run \
  --workflow main_agent --workspace /absolute/run-directory \
  --dangerously-skip-approvals --human-supervised \
  --query "SELF_CONTAINED_CHEMISTRY_TASK" \
  --result-file /absolute/run-directory/result-001.json
```

Add `--config /absolute/config.toml` when configured. MCP tools are unsupported
for new headless sessions. `--human-supervised` enables `ask_human`;
`--dangerously-skip-approvals` independently skips default tool reviews.
Pass the command to `terminal` with `background=true`, `notify=true`, and the
absolute `workdir`. Use the installed version's advertised notification option
or process polling if needed; no PTY is required.

## Monitor and relay

- Record the returned Hermes process handle. Poll/log that process without tight
  loops or relaunching it. Read the early running receipt and immediately record
  its ChemGraph session ID, which differs from the Hermes process handle.
- A running receipt has `exit_code: null`. On process completion, read the final
  receipt and inspect both `exit_code` and `error_code`; a receipt is not proof
  of success. Update `handoff.json` on launch, session discovery, and completion.
- On exit **3**, relay each clarification's `payload.question` to the original
  conversation/user. Match replies to the exact pending IDs and accept them only
  from that user there. Collect every answer, preserving their wording; resolve
  ambiguity without guessing. Tool reviews, unsupported payloads, and missing
  IDs require interactive terminal recovery, as described in the reference.
- On exit **0**, substantiate the response with scientific output files. Preserve
  method, units, convergence, artifact paths, and pending remote-job status.
  Release the active-task association when finished, preserving its record.

## Resume and recover

Write a fresh UTF-8 JSON answers file such as `{"INTERRUPT_ID": "Use EMT."}` with
all current IDs mapped to nonempty strings, then launch and monitor:

```bash
/absolute/chemgraph-env/bin/chemgraph run --workflow main_agent \
  --dangerously-skip-approvals --resume SESSION_ID \
  --answers-file /absolute/run-directory/answers-001.json \
  --result-file /absolute/run-directory/result-002.json
```

Repeat for new questions. Do not combine answers with `--query` or approve tool
reviews through the answers file. A completed session may receive a follow-up
using `--resume SESSION_ID --query "FOLLOW_UP"` and a fresh receipt path.
Saved workspace/configuration takes precedence on resume.

Serialize every operation per session, including inspection. Reconcile the known
process handle before inspection or continuation. A stopped process with only a
running receipt has an uncertain outcome; the ID does not guarantee a checkpoint.
Never automatically replace or replay its calculation. Stop processes only when
requested; external jobs require separate status/cancellation checks.

Read [reference.md](reference.md) for errors, inspection commands, export recovery,
missing IDs, and interactive retry. Inspection uses `--resume` and `--result-file`
without `--query` or `--answers-file`; it does not advance the graph.
