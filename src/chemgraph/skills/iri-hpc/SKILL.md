---
name: iri-hpc
description: Stage, submit, monitor, cancel, and collect ALCF PBS jobs through the IRI Facility API v2 with curl and alcf-tokens. Use for remote ChemGraph calculations on Polaris, Aurora, or Crux when IRI is requested or direct qsub is unavailable.
license: Apache-2.0
compatibility: Requires bash, curl 7.76+, jq, alcf-tokens, globus-cli (through the globus-transfer skill), an ALCF allocation, and an existing compute environment with ChemGraph installed.
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Remote PBS jobs through the IRI API

Use this when the user selects IRI or needs ALCF batch execution from a host that
cannot run `qsub`. If no method was selected and the shell is on a submission host,
prefer [pbs-hpc](../pbs-hpc/SKILL.md). Never mix submission methods for one run. The
JobSpec selects resources, so `#PBS` lines do not apply.

## Setup

Use the user's workspace, or pick one and say so. It holds `target.toml`, run
directories, and transfer records. Use a configuration from elsewhere only when the
user names it; then copy it here (keeping the source) and correct
`local_collection_id` and `local_globus_root` for this workspace.

Run every command from the workspace inside this literal wrapper, even under
DeepAgent's `/bin/sh` (Bash reads the script from fd 3; children get closed stdin).
Skill files may be mounted at a virtual path, so run bundled files from `$SKILLS` and
never retype them:

```bash
exec bash /dev/fd/3 3<<'BASH' </dev/null
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
# Put commands that use SKILLS here, in this same block.
BASH
```

Each block starts fresh. DeepAgent `execute` timeouts: 180 s for submission, 420 s for
IRI filesystem polling, 660 s for Globus wait. Exit 124 can leave a mutation in flight:
inspect saved state per [Resume](references/api-v2.md#resume).

At the start of each session:

1. **Tools and logins.** Check `jq`, `curl` 7.76+, `alcf-tokens test-token iri`
   (`"ready": true`), and `globus whoami`. Tell the user what to install
   (`pip install alcf-tokens globus-cli`) or the exact login to run in their own
   terminal (`alcf-tokens login iri`, `globus login`, or the printed `--gcs` consent).
   A connection failure is not a login problem; see
   [diagnostics](references/api-v2.md#connection-and-authentication).
2. **Target.** Use an existing `./target.toml` as is. Otherwise, for Polaris with
   Eagle storage, run `python "$SKILLS/iri-hpc/scripts/iri_configure.py" .`. It
   discovers IDs, projects, username, and Globus collections, and uses the
   [ChemGraph environment](../pbs-hpc/references/polaris.md#chemgraph-environment)
   unless the user names another (`--environment-setup` and `--python`). Exit 2
   prints JSON questions: ask them all in one message, then rerun with `--project`
   (and `--username`/`--remote-root` if asked). For Aurora, Crux, or non-Eagle roots,
   fill [target.example.toml](assets/target.example.toml) with the user. Never guess
   projects or paths. State the configuration's path, account, and remote
   environment; this is informational, not an approval step.

`target.toml` is the only place IDs live; `scripts/iri.sh` loads them as shell
variables. Never ask for, print, store, or pass a token; send authenticated requests
through the `iri` function in [the recipes](references/api-v2.md).

## Run a calculation

1. **Discover:** compute resource status and the project in `account/projects`.
2. **Write inputs** in a fresh `./$RUN`: the structure (follow
   [local preparation](../chemgraph/references/structure-preparation.md) if absent)
   and `input.json` with run-relative `input_structure_file`/`output_results_file`,
   as in the [ASE batch example](../chemgraph/references/ase-batch.md). Do not install
   the remote calculator locally just to validate the schema.
3. **Prepare:** `python "$SKILLS/iri-hpc/scripts/iri_prepare.py" "$RUN"` writes
   `launch.sh`, `request.json`, and `calculate.py` from `target.toml`. Review its output.
4. **Stage:** IRI has no upload. Follow the [stage recipe](references/api-v2.md#stage):
   IRI `mkdir`, then the `globus-transfer` skill (label `$RUN-stage-in`, records in
   `./.globus/`), wait for `SUCCEEDED`, then IRI `ls` to confirm the inputs.
5. **Submit once:** `python "$SKILLS/iri-hpc/scripts/iri_submit.py" "$RUN"` as its own
   command after staging succeeded. It saves a marker, `job.id`, and `job.json`; report
   the full job ID.
6. **Status and results.** For "submit a job", report the ID and observed state. For
   results, monitor to a terminal state, then collect `job.out`, `job.err`, and the
   result JSON with the [view/tail recipes](references/api-v2.md#collect); move large
   outputs with `globus-transfer` (label `$RUN-collect`) into `$RUN/outputs/`.
   `calculate.py` exits 0 on success, 2 on nonconvergence, 1 on failure; scheduler
   `completed` alone is not scientific success.

Cancel only a job the user asks to cancel, with its full ID
([cancel recipe](references/api-v2.md#cancel)).

## Resume and uncertainty

Accepted jobs survive agent exit. With `job.id`, monitor that job and never resubmit.
With `submission.started` but no ID, stop and follow
[Resume](references/api-v2.md#resume): uncertain submissions are never retried, and an
empty job listing is not permission to resend. Any submission evidence forbids
restaging; never delete evidence or `.submission.lock`, or submit the same
calculation from another directory to work around it. After a definite rejection,
fix the cause (for example in `target.toml`), rerun `iri_prepare.py`, and submit
separately.

For other operations, see the [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/)
and the [API reference](https://api.alcf.anl.gov/).
