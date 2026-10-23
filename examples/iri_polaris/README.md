# EMT water optimization on Polaris through IRI

This example runs a small ChemGraph ASE calculation (`water.xyz`, EMT `opt`) as a
Polaris debug job submitted from a laptop through the ALCF IRI API v2. It uses the
bundled `iri-hpc` skill, so Claude Code, Codex, or a ChemGraph Deep Agent can drive
it with shell commands; no MCP server or ChemGraph agent tool is involved.

## Prerequisites

- An ALCF account, a project with Polaris time, and a compute environment on
  `/eagle` or `/home` with ChemGraph installed.
- On the laptop: `bash`, `curl` 7.76+, `jq`, and `pip install alcf-tokens globus-cli`.
- `alcf-tokens login iri`, then `alcf-tokens test-token iri` shows `"ready": true`.
- Globus Connect Personal running on the laptop, and `globus login`. The IRI API
  cannot upload or download files, so the bundled `globus-transfer` skill stages
  them. `globus ls` on the Eagle collection may print a `globus login --gcs …`
  consent command; run it once.

## Choose an agent

ChemGraph's `deep_agent` and `main_agent` workflows load the bundled skills
automatically. Give them a workspace so they get file and shell tools:

```bash
chemgraph run --interactive -w deep_agent --deepagent-workspace ~/iri-test -m "$LLM_MODEL"
chemgraph run --interactive -w main_agent --workspace ~/iri-test -m "$LLM_MODEL"
```

Launch from the environment where `alcf-tokens` is installed; the agent's shell
inherits only `PATH`, `VIRTUAL_ENV`, and a few other variables. Approve each file
write and shell command, and read the `iri_submit.py` command before approving it.

For Claude Code or Codex, link the installed skills into the agent's skill
directory instead of copying them:

```bash
skills=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
ln -s "$skills/iri-hpc" "$skills/globus-transfer" "$skills/chemgraph" ~/.claude/skills/   # Claude Code
ln -s "$skills/iri-hpc" "$skills/globus-transfer" "$skills/chemgraph" ~/.agents/skills/   # Codex
```

## Run

1. Copy `target.example.toml` from the skill's `assets/` to `~/iri-test/target.toml`
   and fill in your project, remote root, Globus collections, and compute environment.
2. Copy `water.xyz` and `input.json` from this directory to `~/iri-test/run-001/`.
3. Ask the agent: *"Use the iri-hpc skill to run the EMT water optimization in
   run-001 on Polaris with the settings in target.toml. Prepare the remaining
   files, stage them, submit once, and give me the full job ID."*
4. In a new session, ask: *"Continue the Polaris calculation in run-001."* The
   agent reads `job.id` and monitors without submitting again.

A successful run ends with the job in state `completed` and exit code 0, and a
`result.json` containing `"success": true`, `"converged": true`, and a finite
`potential_energy`.
