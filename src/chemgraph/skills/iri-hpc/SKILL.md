---
name: iri-hpc
description: Stage, submit, monitor, cancel, and collect ALCF PBS jobs through the IRI Facility API v2 with curl and alcf-tokens, from a host without SSH or qsub. Use for remote ChemGraph calculations on Polaris, Aurora, or Crux via IRI.
license: Apache-2.0
compatibility: Requires bash, curl 7.76+, jq, alcf-tokens, globus-cli (through the globus-transfer skill), an ALCF allocation, and an existing compute environment with ChemGraph installed.
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Remote PBS jobs through the IRI API

Use this workflow when the user wants ALCF batch execution from a host that cannot
run `qsub`. On a login node, use the `pbs-hpc` skill directly instead. Never mix
submission methods for one run. Read the `pbs-hpc` site reference for environment
and launch requirements; `#PBS` directives do not apply here because the JobSpec
selects resources.

## Setup

The user only describes the calculation; set up the rest. Run every command from
the workspace that holds the run directories and `target.toml` (in Claude Code or
Codex, start each command with `cd` to it). The recipes need bash: where the
agent's shell is `sh` or `dash` (common on Linux), run each command as
`bash -c '…'`. Skill files may be mounted at a virtual
path such as `/chemgraph-skills/`, while shell commands run on the host, so resolve
the installed skills directory and run bundled files from it; never retype them:

```bash
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
```

At the start of each session:

1. **Tools and logins.** Check `command -v jq curl`, that `curl --version` is 7.76
   or newer, `alcf-tokens test-token iri` (`"ready": true`), and `globus whoami`.
   Tell the user what to install if a tool is missing (`jq`, `curl`,
   `pip install alcf-tokens globus-cli`). If a login fails, stop and give the user
   the exact command to run in their own terminal: `alcf-tokens login iri` (needed
   again every 7 days), `globus login`, or the `globus login --gcs …` consent the CLI
   printed.
2. **Target.** If `./target.toml` is missing:
   - For Polaris (the default) with Eagle storage, run
     `python "$SKILLS/iri-hpc/scripts/iri_configure.py" .`. It discovers resource
     IDs, the user's projects and username, and the Globus collections. Exit 2
     prints JSON questions: ask the user all of them in one message, then rerun with
     `--project`, `--environment-setup`, and `--python` (plus `--username` or
     `--remote-root` when asked or wanted).
   - For Aurora, Crux, or a remote root outside Eagle, copy
     [target.example.toml](assets/target.example.toml) to `./target.toml` and fill it
     with the user, taking resource IDs from `status/resources` by name.
   Never guess projects, paths, or environments.

`target.toml` is the only place IDs live. Sourcing `scripts/iri.sh` loads them as
shell variables; never retype IDs into commands. Never ask for, print, store, or pass
a token as a command argument; send every authenticated request through the `iri`
function, as shown in [the v2 recipes](references/api-v2.md).

## Run a calculation

1. **Discover.** Check the compute resource status and that the project appears in
   `account/projects`.
2. **Prepare** a fresh run directory `./$RUN`; its remote counterpart is
   `$REMOTE_ROOT/$RUN`. One run name drives every path:
   - the structure file (follow the `chemgraph` skill's structure preparation if absent);
   - `input.json` with run-relative `input_structure_file` and `output_results_file`;
   - the `chemgraph` skill's [calculate.py](../chemgraph/assets/calculate.py), copied
     unchanged with `cp "$SKILLS/chemgraph/assets/calculate.py" "$RUN/"`;
   - `launch.sh` from [the launch template](assets/launch.sh.template), checked with `bash -n`;
   - `request.json`: `{"resource_id": ..., "job_spec": ...}` as in the recipes.
3. **Stage.** ALCF's IRI API has no upload or download. Create the remote
   directory with IRI `mkdir`, then transfer `./$RUN` with the `globus-transfer`
   skill, labelled `$RUN-stage-in`, with records in `./.globus/`, from
   `$LOCAL:$LOCAL_ROOT/$RUN/` to `$REMOTE_COLLECTION:$REMOTE_GLOBUS_ROOT/$RUN/` as in
   the stage recipe. Wait until that task is `SUCCEEDED`, then confirm with IRI `ls`
   that input dependencies (structure, model files) exist. Output files are created
   by the job.
4. **Submit once** with `python "$SKILLS/iri-hpc/scripts/iri_submit.py" "$RUN"`,
   as its own command after the stage-in task succeeded and you reviewed the staged
   listing; never chain it after staging.
   It records `submission.started`, then `job.id` and `job.json`. Report the full
   job ID.
5. **Monitor** with the job status recipe until a terminal state.
6. **Cancel** only a job the user asks to cancel, using its full ID.
7. **Collect** `job.out`, `job.err`, and the result JSON with the IRI `view` or
   `tail` recipes, and save them in the local run directory. Move large or binary
   outputs with `globus-transfer` (label `$RUN-collect`) into `$RUN/outputs/`.
8. **Interpret** with the `chemgraph` skill: `calculate.py` exits 0 for success,
   2 for nonconvergence, and 1 for failure. Scheduler `completed` alone does not
   establish scientific success; early failures may leave only stderr.

## Resume and uncertainty

A later session starts from the saved evidence in the run directory:

- `job.id` exists: monitor and collect that job. Never submit again.
- `submission.started` exists without `job.id`: the outcome is uncertain. Stop and
  report it. Inspect `submission.uncertain.json` and list all of the user's jobs
  with the paged recipe; IRI returns no job names, so show the user each job's
  account, queue, and state. If the user confirms none is this run and approves, run `iri_submit.py --retry-uncertain
  "$RUN"` with the same script; it resends the identical request with its original idempotency key.
- `submission.rejected.json` without a marker: the initial request was rejected.
  Fix the cause (for a 401, ask the user to log in again) and submit again.
- No submission marker, but a `.globus/$RUN-stage-in.json` record:
  resume that transfer with the `globus-transfer` skill; never start another.

Never delete `submission.started`, edit a submitted `request.json`, or resubmit in
another directory to work around uncertainty. Accepted jobs survive agent exit.

For other operations, consult the [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/)
and the [API reference](https://api.alcf.anl.gov/).
