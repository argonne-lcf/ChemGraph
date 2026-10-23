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

## Authenticate and configure

The user authenticates in a terminal: `pip install alcf-tokens`, then
`alcf-tokens login iri`. Check with `alcf-tokens test-token iri` (`"ready": true`).
Never ask for, print, store, or pass a token as a command argument; send every
authenticated request through the `iri` function from `scripts/iri.sh`, as shown in
[the v2 recipes](references/api-v2.md).

Copy [target.example.toml](assets/target.example.toml) next to the run directories
and fill it with the user's project, queue, remote root, Globus collections, and
compute environment.
Ask for missing values; never guess projects, paths, or environments.

## Run a calculation

Skill files may be mounted at a virtual path such as `/chemgraph-skills/`, while
shell commands run on the host. Resolve the installed skills directory first and
copy or run bundled files from it; never retype them:

```bash
SKILLS=$(python -c "import chemgraph.skills, pathlib; print(pathlib.Path(chemgraph.skills.__file__).parent)")
```

1. **Discover.** Check the compute resource status and that the project appears in
   `account/projects`.
2. **Prepare** a fresh local run directory with a fresh remote counterpart below
   `remote_root`:
   - the structure file (follow the `chemgraph` skill's structure preparation if absent);
   - `input.json` with run-relative `input_structure_file` and `output_results_file`;
   - the `chemgraph` skill's [calculate.py](../chemgraph/assets/calculate.py), copied
     unchanged with `cp "$SKILLS/chemgraph/assets/calculate.py" RUN_DIR/`;
   - `launch.sh` from [the launch template](assets/launch.sh.template), checked with `bash -n`;
   - `request.json`: `{"resource_id": ..., "job_spec": ...}` as in the recipes.
3. **Stage.** ALCF's IRI API has no upload or download. Create the remote
   directory with IRI `mkdir`, then transfer the run directory with the
   `globus-transfer` skill, labelled `<run>-stage-in`, with records in
   `<workspace>/.globus/`. The destination is the same run directory below
   `remote_globus_root`, the collection's name for `remote_root`. Wait until that task is `SUCCEEDED`, then confirm with
   IRI `ls` that input dependencies (structure, model files) exist. Output files
   are created by the job.
4. **Submit once** with `python "$SKILLS/iri-hpc/scripts/iri_submit.py" RUN_DIR`,
   as its own command after the stage-in task succeeded and you reviewed the staged
   listing; never chain it after staging.
   It records `submission.started`, then `job.id` and `job.json`. Report the full
   job ID.
5. **Monitor** with the job status recipe until a terminal state.
6. **Cancel** only a job the user asks to cancel, using its full ID.
7. **Collect** `job.out`, `job.err`, and the result JSON with the IRI `view` or
   `tail` recipes, and save them in the local run directory. Move large or binary
   outputs with `globus-transfer` (label `<run>-collect`) into `RUN_DIR/outputs/`.
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
  RUN_DIR` with the same script; it resends the identical request with its original idempotency key.
- `submission.rejected.json` without a marker: the initial request was rejected.
  Fix the cause (for a 401, ask the user to log in again) and submit again.
- No submission marker, but a `<workspace>/.globus/<run>-stage-in.json` record:
  resume that transfer with the `globus-transfer` skill; never start another.

Never delete `submission.started`, edit a submitted `request.json`, or resubmit in
another directory to work around uncertainty. Accepted jobs survive agent exit.

For other operations, consult the [ALCF IRI guide](https://docs.alcf.anl.gov/services/iri-api/)
and the [API reference](https://api.alcf.anl.gov/).
