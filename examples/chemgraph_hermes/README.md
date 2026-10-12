# Delegate chemistry tasks from Hermes to ChemGraph

Hermes launches the ChemGraph CLI, monitors its process, and reads its results.
ChemGraph owns the scientific workflow and saves its own sessions. This personal,
local setup needs a Hermes skill and an existing ChemGraph installation; no
Connect service, MCP server, or Hermes plugin is required.

## Setup

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
See the [skill](skills/chemgraph-delegate/SKILL.md) for exit codes, export recovery,
and clarification handling, and the [CLI guide](../../docs/cli.md) for details.
