# EMT water optimization on Polaris through IRI

This example runs a small ChemGraph ASE calculation (`water.xyz`, EMT `opt`) as a
Polaris debug job submitted from a laptop through the ALCF IRI API v2. It uses the
bundled `iri-hpc` skill, so Claude Code, Codex, or a ChemGraph Deep Agent can drive
it with shell commands; no MCP server or ChemGraph agent tool is involved.

## Prerequisites

- An ALCF account, a project with Polaris time, and a compute environment on
  `/eagle` or `/home` with ChemGraph installed.
- On the laptop: `bash`, `curl` 7.76+, `jq`, and `pip install chemgraph alcf-tokens globus-cli`
  in the Python environment the agent's shell commands use (they locate the bundled
  skills with `import chemgraph.skills`).
- `alcf-tokens login iri`, then `alcf-tokens test-token iri` shows `"ready": true`.
- Globus Connect Personal running on the laptop, and `globus login`. The IRI API
  cannot upload or download files, so the bundled `globus-transfer` skill stages
  them. `globus ls` on the Eagle collection may print a `globus login --gcs …`
  consent command; run it once.

## Choose an agent

ChemGraph's `deep_agent` and `main_agent` workflows load the bundled skills
automatically. Create and select a workspace so they get file and shell tools:

```bash
mkdir -p ~/iri-workspace
cd ~/iri-workspace
# Choose one workflow:
chemgraph run --interactive -w deep_agent --deepagent-workspace "$PWD" -m "$LLM_MODEL"
chemgraph run --interactive -w main_agent --workspace "$PWD" -m "$LLM_MODEL"
```

On later sessions, return to this workspace without recreating it. `~/iri-workspace`
is an illustrative choice, not a location for agents to search for configuration.
Keep the workspace under your home directory: Globus Connect Personal shares home
by default, and `iri_configure.py` maps it as `/~/...`.

Launch from the environment where `alcf-tokens` and `globus` are installed; the agent's shell
inherits only `PATH`, `VIRTUAL_ENV`, and a few other variables. Follow the host's
actual tool approvals; this workflow does not add a separate approval for every
file write or shell command. Review submission details before the submission step;
when approvals are on, read the `iri_submit.py` command before approving it.
Use the skills' complete Bash recipe blocks, including the fd 3 wrapper and variable
setup. For DeepAgent, allow 180 seconds for submission, 420 for IRI filesystem
polling, and 660 for Globus wait. Exit 124 means inspect saved state and wait; it
never authorizes replaying a job submission.

For Claude Code or Codex, link the installed skills into the agent's skill
directory instead of copying them:

```bash
skills=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
ln -s "$skills/iri-hpc" "$skills/globus-transfer" "$skills/chemgraph" ~/.claude/skills/   # Claude Code
ln -s "$skills/iri-hpc" "$skills/globus-transfer" "$skills/chemgraph" ~/.agents/skills/   # Codex
```

## Run

From the selected workspace, ask: *"Optimize a water molecule with EMT on Polaris
and return the results."* The normal sequence is:

1. **Configure.** Check this workspace's `target.toml`. If absent, the agent uses
   `iri_configure.py` for Polaris with Eagle storage, discovers resource IDs,
   projects, username, and collections where possible, and asks its unresolved
   questions together. The project and existing remote `environment_setup` and
   `python` come from you or your deployment configuration. The remote environment
   must already have ChemGraph and the calculator installed; it is not created by
   this workflow. The agent states the configuration source, account, and remote
   environment in use. Later sessions reuse this workspace's file. Use another
   configuration only when explicitly identified, and correct its local Globus
   workspace mapping when importing it.
2. **Prepare.** Create a fresh local run directory, generate and check the water
   structure, and write the calculation and launch inputs. `input.json` uses
   run-relative paths such as `water.xyz` and `result.json`; local absolute paths
   do not refer to Polaris files.
3. **Stage.** Transfer the run directory with Globus, wait for `SUCCEEDED`, and
   inspect the remote inputs before submission.
4. **Submit.** Run `iri_submit.py` once as a separate command and save the full
   job ID and submission records.
5. **Monitor and collect as requested.** For “submit the job,” report the accepted
   ID and observed status. For calculated results, monitor to a terminal state,
   collect output, and check the result JSON. In a later session, ask *"How's my
   water optimization on Polaris going?"* from the same workspace; the agent reads
   `job.id` and monitors without submitting again.

`water.xyz` and `input.json` in this directory show what the agent prepares.

An accepted or queued job is a submission receipt, not successful calculation
results. A successful run ends with the job in state `completed` and exit code 0,
and a `result.json` containing `"success": true`, `"converged": true`, and a finite
`potential_energy`.

The helper submits without `Idempotency-Key`, so an ALCF deployment without an
idempotency store can accept the job. Its local marker prevents an uncertain
request from being resent; `--retry-uncertain` is disabled. Preserve uncertain
runs and contact the facility operator. An empty job listing is not proof of rejection.

For an old run that failed with the exact missing-idempotency-store 501, see the
[legacy recovery recipe](../../src/chemgraph/skills/iri-hpc/references/api-v2.md#resolve-a-legacy-idempotency-rejection).
Recovery requires the user's confirmation after reviewing **every** prior attempt,
archives that evidence, and submits nothing. Then submit separately without
restaging. A newer headerless 501 is not eligible for that recovery.
