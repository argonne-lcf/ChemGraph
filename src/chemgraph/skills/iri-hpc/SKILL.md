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
run `qsub`. On an authorized submission host, use [pbs-hpc](../pbs-hpc/SKILL.md)
directly instead. Never mix submission methods for one run. Use its relevant
site-reference sections for environment and launch requirements; `#PBS`
directives do not apply here because the JobSpec selects resources.

## Setup

First identify the local workspace: use the user's selected workspace, or select
and state one for this task. It holds `target.toml`, run directories, and transfer
records. Check **that workspace's** `target.toml`; use a configuration elsewhere
only when the user identifies it. Example paths are illustrative, not locations
to search for configuration. If importing a configuration, place a copy at the
selected workspace's `target.toml` and correct `local_collection_id` and
`local_globus_root` to map this workspace, rather than staging the old workspace.
Preserve the source configuration.

Run every command from the selected workspace (in Claude Code or Codex, start
each command with `cd` to it). Use the literal Bash wrapper in the
recipes, including under DeepAgent's `/bin/sh`: Bash reads the script from fd 3
and child commands receive closed stdin. This requires Bash and `/dev/fd` support;
if either is unavailable, stop and report it. Skill files may be mounted at a virtual
path such as `/chemgraph-skills/`, while shell commands run on the host, so resolve
the installed skills directory and run bundled files from it; never retype them:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
# Put setup commands that use SKILLS here, in this same block.
BASH
```

Each execution starts fresh: include its own variables and setup inside the wrapper.
For DeepAgent `execute`, use `timeout=180` for submission, `timeout=420` for IRI
filesystem polling, and `timeout=660` for Globus wait. Exit 124 can leave a mutation
in flight: inspect saved state and follow [resume guidance](references/api-v2.md#resume);
never replay it blindly.

At the start of each session:

1. **Tools and logins.** Check `command -v jq curl`, that `curl --version` is 7.76
   or newer, `alcf-tokens test-token iri` (`"ready": true`), and `globus whoami`.
   Tell the user what to install if a tool is missing (`jq`, `curl`,
   `pip install alcf-tokens globus-cli`). For confirmed missing/expired credentials
   or required consent, give the user the exact command to run in their own
   terminal: `alcf-tokens login iri`, `globus login`, or the `globus login --gcs …`
   consent the CLI printed. Connection failures alone do not establish a login
   problem; use [the diagnostics](references/api-v2.md#connection-and-authentication).
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
3. **State the configuration in use.** Identify its source path (including the
   original source if imported), account, and remote `environment_setup` and
   `python`. These describe an existing remote environment supplied by the user
   or deployment, not a laptop environment or an environment to install
   automatically. This is informational, not another approval step.

`target.toml` is the only place IDs live. Sourcing `scripts/iri.sh` loads them as
shell variables; never retype IDs into commands. Never ask for, print, store, or pass
a token as a command argument; send every authenticated request through the `iri`
function, as shown in [the v2 recipes](references/api-v2.md).

## Run a calculation

1. **Discover.** Check the compute resource status and that the project appears in
   `account/projects`.
2. **Prepare** a fresh run directory `./$RUN`; its remote counterpart is
   `$REMOTE_ROOT/$RUN`. One run name drives every path:
   - the structure file (follow
     [local preparation](../chemgraph/references/structure-preparation.md) if absent);
   - `input.json` with run-relative `input_structure_file` and `output_results_file`;
   - the `chemgraph` skill's [calculate.py](../chemgraph/assets/calculate.py), copied
     unchanged with `cp "$SKILLS/chemgraph/assets/calculate.py" "$RUN/"`;
   - `launch.sh` from [the launch template](assets/launch.sh.template), checked with `bash -n`;
   - `request.json`: `{"resource_id": ..., "job_spec": ...}` as in the recipes.
   Use [ASE input checks](../chemgraph/references/ase-batch.md) without requiring
   the remote calculator's extras on the laptop solely for schema validation.
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
   The helper sends no `Idempotency-Key` (ALCF may have no idempotency store). It
   records a local attempt marker and history, then `job.id` and `job.json`. Report
   the full job ID. Without server deduplication, uncertain submissions cannot be retried.
5. **Check status.** For “submit a job,” report the accepted ID and observed state,
   including queued or held. For completed results, monitor to a terminal state;
   if reporting while pending, retain the ID and say results are outstanding.
6. **Collect when results are requested:** `job.out`, `job.err`, and the result JSON
   with the IRI `view` or `tail` recipes, and save them in the local run directory.
   Move large or binary outputs with `globus-transfer` (label `$RUN-collect`) into
   `$RUN/outputs/`.
7. **Interpret** with the `chemgraph` skill: `calculate.py` exits 0 for success,
   2 for nonconvergence, and 1 for failure. Scheduler `completed` alone does not
   establish scientific success; early failures may leave only stderr.

Cancel only a job the user asks to cancel, using its full ID and the
[cancel recipe](references/api-v2.md#cancel).

## Resume and uncertainty

Resume from saved evidence; accepted jobs survive agent exit. With `job.id`,
monitor that job without resubmitting. With `submission.started` but no ID, stop
and follow [resume and uncertainty procedures](references/api-v2.md#resume).
`--retry-uncertain` always refuses; an empty job listing never permits a resend.

Any `submission.started`, `job.id`, or `submission.history` prohibits stage-in.
Never delete evidence or `.submission.lock`, edit an uncertain request, or submit
in another directory to work around uncertainty. Resume an existing transfer from
its saved record. Definite rejections require correction and a separate submit.
The only legacy recovery is the
[exact missing-idempotency-store rejection](references/api-v2.md#resolve-a-legacy-idempotency-rejection),
with evidence for **every** prior attempt and the user's confirmation. Recovery
submits nothing; any later submission is separate and never restages the run.

For other operations, consult the [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/)
and the [API reference](https://api.alcf.anl.gov/).
