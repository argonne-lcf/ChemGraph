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
automatically. Give them a workspace so they get file and shell tools:

```bash
chemgraph run --interactive -w deep_agent --deepagent-workspace ~/iri-test -m "$LLM_MODEL"
chemgraph run --interactive -w main_agent --workspace ~/iri-test -m "$LLM_MODEL"
```

Launch from the environment where `alcf-tokens` and `globus` are installed; the agent's shell
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

1. Start from an empty workspace, for example `mkdir ~/iri-test`, and launch the
   agent there.
2. Ask: *"Optimize a water molecule with EMT on Polaris."*
3. The first time, the agent writes `target.toml` itself. It discovers the
   resource IDs, your projects and ALCF username, and the Globus collections, then
   asks you once for what it cannot know: the project (if you have several) and
   the Polaris Python environment that has ChemGraph installed. Later sessions
   reuse that file.
4. The agent prepares the run directory, stages it with Globus, submits once, and
   reports the full job ID. Approve the `iri_submit.py` command after reading it.
5. In a new session, ask: *"How's my water optimization on Polaris going?"* The
   agent reads `job.id` and monitors without submitting again.

`water.xyz` and `input.json` in this directory show what the agent prepares.

A successful run ends with the job in state `completed` and exit code 0, and a
`result.json` containing `"success": true`, `"converged": true`, and a finite
`potential_energy`.
