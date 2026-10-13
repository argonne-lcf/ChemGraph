# Delegate chemistry tasks from Hermes to ChemGraph

Hermes launches the ChemGraph CLI, monitors its process, and reads its results.
ChemGraph owns the scientific workflow and saves its own sessions. This personal,
local setup needs a Hermes skill and an existing ChemGraph installation; no
Connect service, MCP server, or Hermes plugin is required.
The receipt is a CLI contract. This example does not provide the trusted identity,
task authorization, or exact-action approvals required by the Connect service.

## Setup

Run Hermes and ChemGraph under a dedicated unprivileged OS user or in a container
with only the inputs, output directories, and credentials needed for these tasks.
Configure the active gateway profile with an explicit numeric user allowlist,
for example `DISCORD_ALLOWED_USERS=YOUR_DISCORD_USER_ID`. Disable allow-all
settings and other grants that would admit unintended users; verify denial from
a non-allowlisted account. See the [Hermes Discord access controls](https://hermes-agent.nousresearch.com/docs/user-guide/messaging/discord/).
Accepted users can direct the enabled shell tools with that OS account's
privileges. The skill's conversation/user checks and `handoff.json` are operating
instructions, not an authentication or isolation boundary.

Use a ChemGraph checkout whose `run --help` lists `--result-file` and
`--answers-file`. From its root, install into a separate environment, replacing
the example path:

```bash
python3 -m venv /absolute/chemgraph-env
/absolute/chemgraph-env/bin/python -m pip install -e .
/absolute/chemgraph-env/bin/chemgraph run --help
```

Configure the [model and credentials](../../docs/models.md) for that environment.
If using a [TOML configuration](../../docs/configuration_with_toml.md), pass its
absolute path with `--config`; this example cannot use a configured MCP server.
Install any calculators your task needs in the ChemGraph environment.

Install the skill from the repository root into your **active Hermes profile**:

```bash
hermes_profile_dir=/absolute/path/to/active-hermes-profile
mkdir -p "$hermes_profile_dir/skills"
cp -R examples/chemgraph_hermes/skills/chemgraph-delegate "$hermes_profile_dir/skills/"
```

The default profile normally uses `~/.hermes`; choose the actual profile directory
for a named profile or custom `HERMES_HOME`. Start a new Hermes conversation with
terminal, process, file, and skill tools available. See the Hermes
[skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills) and
[tools](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools) guides.

Tell Hermes the absolute ChemGraph executable, optional config path, and parent
directory for runs. For example: “Use chemgraph-delegate to build water from
SMILES O, optimize it with EMT as a demonstration, and report its energy and
convergence. Use /absolute/chemgraph-env/bin/chemgraph and /absolute/cg-runs.”

## Execution and continuation

This example opts into `--dangerously-skip-approvals` for trusted personal work.
It skips default tool reviews and permits host shell access with `--workspace`;
the workspace is not a sandbox. Genuine questions and stricter policies can pause.
`--human-supervised` enables `ask_human` for clarifications; approval bypass
independently skips default tool reviews.

Hermes creates a fresh run directory and launches this command in the background:

```bash
CHEMGRAPH_LOG_DIR=/absolute/cg-runs/task-001 \
/absolute/chemgraph-env/bin/chemgraph run \
  --workflow main_agent --workspace /absolute/cg-runs/task-001 \
  --dangerously-skip-approvals --human-supervised \
  --query "Build water from SMILES O and optimize it with EMT; report convergence and energy." \
  --result-file /absolute/cg-runs/task-001/result-001.json
```

Hermes monitors its process handle, reads the receipt's `assistant_response`,
and inspects scientific artifacts. The receipt's `session_id` identifies the
ChemGraph session; it is separate from the Hermes process handle.
Before graph execution, the receipt has `status: "running"` and `exit_code: null`.
The final receipt records the exit code and any `error_code`/`error`. If the
initial receipt cannot be written, the graph does not run.

## Questions through Discord

On exit **3**, the receipt contains pending interrupts. For a `clarification`,
Hermes sends `payload.question` to the requesting user in the original Discord
conversation. When that user replies, Hermes writes their answer against the
exact interrupt ID in a fresh file, for example `answers-001.json`:

```json
{"INTERRUPT_ID_FROM_RESULT": "Use EMT for this demonstration."}
```

It then starts a new background process for the same ChemGraph session:

```bash
/absolute/chemgraph-env/bin/chemgraph run --workflow main_agent \
  --dangerously-skip-approvals --resume SESSION_ID \
  --answers-file /absolute/cg-runs/task-001/answers-001.json \
  --result-file /absolute/cg-runs/task-001/result-002.json
```

If more questions appear, repeat with their new IDs. Gather answers to all pending
questions before submitting. `--resume ... --query ...` starts a new turn after
completion; it does not answer a pending question.

Hermes maintains one active task per Discord conversation and requesting user.
The skill records that association in the task's `handoff.json`, together with
the workspace, ChemGraph session ID, process handle, receipt path, and pending
IDs. ChemGraph owns checkpoints and validates answer IDs. Hermes owns Discord
messages, routing, and process monitoring. The skill is an operating procedure;
there is no new Discord router or background service enforcing it.

After a missed notification or restart, inspect without running tools again:

```bash
/absolute/chemgraph-env/bin/chemgraph run --workflow main_agent \
  --dangerously-skip-approvals --resume SESSION_ID \
  --result-file /absolute/cg-runs/task-001/inspection-001.json
```

First reconcile the known process handle and confirm it has stopped. A leftover
running receipt means an uncertain outcome; its session ID does not prove that
a checkpoint was saved. Never automatically repeat the calculation to recover
from a missing final receipt. Serialize inspection and execution per session.

Use a new receipt filename on every invocation. For a completed session's
follow-up, add `--query "Follow-up task"`. Tool reviews and unsupported interrupts
still require a terminal:

```bash
/absolute/chemgraph-env/bin/chemgraph run --interactive --workflow main_agent \
  --dangerously-skip-approvals --resume SESSION_ID
```

Saved sessions/checkpoints support recovery, but unfinished processes are not
guaranteed to survive a crash. A finished agent turn does not prove convergence
or completion of remote jobs; stopping its process does not cancel those jobs.
See the [skill](skills/chemgraph-delegate/SKILL.md) for the workflow, its
[reference](skills/chemgraph-delegate/reference.md) for errors and recovery,
and the [CLI guide](../../docs/cli.md) for the receipt contract.

## Live acceptance checklist

Run this against the changed checkout before merging the example. Record its
`git rev-parse HEAD`, the Hermes version/profile, OS, and model ID. Update the
ChemGraph environment and copy the skill directory using the setup commands
above, then start a fresh Hermes conversation. Use a disposable run directory
and an allowlisted Discord test conversation. Keep credentials out of the record.

Send this task through Discord, substituting your executable and run parent:

> Use chemgraph-delegate with /absolute/chemgraph-env/bin/chemgraph and
> /absolute/cg-acceptance. Have ChemGraph call ask_human to ask me which calculator
> to use before doing any calculation. After my reply, prepare two copper atoms
> 2.6 angstrom apart, optimize with at most 10 steps and fmax 0.05 eV/angstrom,
> and report the calculator, energy in eV, convergence, and output paths.
> This is an integration demonstration, not a research result. Do not choose
> the calculator or answer the clarification for me.

At the clarification, verify that the process exited **3**, the receipt has
`status: "waiting_for_user"`, `error_code: null`, and a clarification with a
nonempty ID, and the question reached the requesting Discord conversation.
Before replying, inspect the same session from the terminal. Substitute the
actual task directory and receipt filename from Hermes:

```bash
cg=/absolute/chemgraph-env/bin/chemgraph
cg_python=/absolute/chemgraph-env/bin/python
task_dir=/absolute/cg-acceptance/TASK_DIRECTORY
first_receipt="$task_dir/result-001.json"
session_id=$("$cg_python" -c 'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["session_id"])' "$first_receipt")
"$cg" run -w main_agent --dangerously-skip-approvals \
  --resume "$session_id" --result-file "$task_dir/inspection-001.json"
```

Expect exit **3**, identical pending IDs, and no model/tool execution. While the
session is still paused, verify stale-answer rejection:

```bash
printf '%s\n' '{"stale-id": "EMT"}' > "$task_dir/stale-answers.json"
"$cg" run -w main_agent --dangerously-skip-approvals --resume "$session_id" \
  --answers-file "$task_dir/stale-answers.json" \
  --result-file "$task_dir/rejected-stale.json"
```

Expect exit **2**, `error_code: "stale_answers"`, and unchanged pending IDs.
Then send “Use EMT for this demonstration” through Discord. Verify that Hermes writes
answers keyed by the current IDs, uses the same ChemGraph session ID and a fresh
receipt path, and returns the result to the same conversation. Answer any further
clarifications explicitly. The final receipt must have exit **0**, completed
status, and null errors. Check the actual output files for energy units, method,
and convergence; reaching the step limit must be reported accurately.

After completion, submit the same test file again:

```bash
"$cg" run -w main_agent --dangerously-skip-approvals --resume "$session_id" \
  --answers-file "$task_dir/stale-answers.json" \
  --result-file "$task_dir/rejected-not-waiting.json"
```

Expect exit **2**, `error_code: "not_waiting"`, and completed status retained.

Also exercise these cases, using fresh filenames for every invocation:

| Case | Expected outcome |
| --- | --- |
| Non-allowlisted account messages the bot | Gateway denies access; no ChemGraph task starts |
| Restart Hermes while waiting, then request inspection | Same saved session/questions; no replacement calculation |
| Inspect completed session | Exit **0**; no repeated calculation or output-file modifications |

Run the automated process-termination check from the checkout in a test-enabled
environment: `python -m pytest tests/test_headless_handoff.py::test_sigterm_leaves_running_receipt -q`.
It terminates a real child process with a blocked model stub and verifies that
the running receipt survives. For a separate disposable live task, the same
expectation applies if you terminate its known CLI process after the running
receipt appears: retain the session ID, reconcile any external jobs, and inspect
without automatic retry.

Keep the receipts, handoff record, process exit codes, and a redacted Discord
transcript as acceptance evidence. Automated CLI tests do not establish live
gateway routing or notification behavior.
