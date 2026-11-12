---
name: pbs
description: Prepare, submit, inspect, modify, cancel, and recover PBS jobs using native shell commands on the selected machine.
license: Apache-2.0
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# PBS jobs through shell access

Use native commands on the target submission host. Obtain account, queue,
resources, filesystem, and environment settings from the selected machine
reference, user configuration, or inspected environment. Use software guidance
for application setup and result validation. Preparation alone does not authorize
submission; an instruction to run the workload does.

## Prepare

Verify local resource syntax and limits, then adapt existing scripts to the
workload. Put `#PBS` directives before executable statements, replace
placeholders, and reconcile command-line overrides. Directives do not expand
shell variables. With `select`, distinguish per-chunk resources from job-wide
walltime; chunks are not universally physical nodes. Match application
threads/ranks and memory to the allocation.

Choose the submission directory explicitly. In the job, use
`cd -- "$PBS_O_WORKDIR" || exit 1` when that is the intended working directory.
Check compute-node access to inputs, software, output directories, and scratch;
stage results out before temporary storage is reclaimed. Run `bash -n job.pbs`
for Bash syntax checks; this does not validate PBS directives. Preserve payload
exit codes through pipelines and cleanup.

## Submit

Replace uppercase placeholders with verified settings:

```bash
cd /absolute/submission/directory || exit 1
qsub -A PROJECT -q QUEUE job.pbs
```

Record the full returned ID, submission time/host/directory, script version,
effective options, and command result. Confirm accepted settings with
`qstat -f JOB_ID`.

Verify local support before using these patterns; supply other required site
settings in the scripts or command:

```bash
qsub -W depend=afterok:JOB_ID analysis.pbs
qsub -J START-END array.pbs
```

Array syntax (`-J` or alternatives), index variables, concurrency limits, and
dependency types vary. Use the documented index variable, map indices to inputs,
and give subjobs distinct output/scratch paths. `afterok` checks exit status;
validate application results in the predecessor if downstream work requires it.

If submission times out or returns no clear ID, inspect the queue and supported
history before retrying. Match owner, name, time, paths, and resources; a name
alone is insufficient. Resolve an ambiguous outcome before submitting another
copy or cancelling a candidate.

## Inspect

```bash
qstat -u "$USER"
qstat -f JOB_ID
qstat -Qf
```

Check installed manuals or site documentation before using JSON
(`qstat -f -F json JOB_ID`) or finished-job history (`qstat -x -f JOB_ID`).
History must also be enabled and retained; otherwise use available text output
and accounting records. Quote array IDs to prevent shell glob expansion.
Avoid exposing `Variable_List`, which can contain credentials.

Common states: Q queued, R running, H held, E exiting. Finished states such as
F or C depend on the implementation. Inspect comments, holds, dependencies,
requested/used resources, exit status, stdout/stderr, and application outputs.
A missing job does not establish completion. Interpret exit codes using local
PBS documentation and payload logs. Even exit status zero does not establish
application correctness or scientific convergence.

## Modify or cancel

```bash
qdel JOB_ID
qhold JOB_ID
qrls JOB_ID
qalter -l walltime=HH:MM:SS JOB_ID
```

Use identified jobs within the requested scope; distinguish array parents from
subjobs. `qdel` can terminate running jobs. `qhold` sets a user hold, not a process
suspension; `qrls` releases it. Permitted `qalter` changes depend on site policy
and job state. Capture the response and recheck with `qstat`.

## Recover and report

Preserve logs/checkpoints and diagnose resource, environment, filesystem, or
application failures. Follow software restart rules and confirm the earlier
attempt is no longer active before an authorized replacement. Record retries;
stop when the same unresolved failure recurs. Report job IDs, paths, observed
states, exit codes, outputs, and unresolved issues.
