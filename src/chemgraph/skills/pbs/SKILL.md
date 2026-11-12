---
name: pbs
description: Prepare, submit, inspect, modify, cancel, and recover PBS jobs using native shell commands on the selected machine.
license: Apache-2.0
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# PBS jobs through shell access

Use native PBS commands through the agent's shell tool on the target submission
host. This skill is standalone: no Python wrapper, backend, receipt schema, or
fixed script template is required. Obtain machine settings from the selected
machine reference, user configuration, or inspected environment; obtain payload
setup and result-validation rules from the relevant software guidance.

## Establish the environment and scope

Confirm the target host, scheduler implementation, user, working directory, and
requested action. Inspect existing scripts and local command manuals/help.
Resolve the project, queue, resource syntax, limits, filesystem visibility, and
software environment before submission; do not invent defaults. Ask only for
missing essential information.

Preparation or inspection does not authorize submission. An instruction to run
the requested workload does; proceed within that scope without repeatedly asking
for approval. Control only the identified jobs covered by the user's request.

Common PBS commands are listed below, but options and policies vary by
implementation and version. Verify optional features using installed manuals,
site documentation, or read-only queries, never trial submissions:

| Feature | Check before use |
| --- | --- |
| Resource requests | Whether `select` chunks or another syntax are supported; memory scope, CPU/rank meaning, placement, and limits |
| Arrays | `-J` versus other array syntax, index variable, throttling, limits, subjob IDs, and dependency support |
| Structured output | Whether `qstat -f -F json JOB_ID` is supported; otherwise use text |
| Finished-job history | Whether `qstat -x -f JOB_ID` is supported and history is enabled/retained; a missing record is inconclusive |
| Dependencies and changes | Supported dependency types and changes permitted for the job's current state |

## Prepare scripts and submission directories

Adapt the user's script to the task. Place `#PBS` directives before executable
statements. Replace all placeholders before use; directives are parsed by PBS,
so do not rely on shell variable expansion inside them. Reconcile command-line
overrides with directives and record the effective settings.

Choose resources from workload needs and site limits. For implementations with
`select`, distinguish per-chunk resources from job-wide walltime; chunks are not
universally physical nodes. Match application threads/ranks and memory to the
allocation. Set output/error destinations that exist and are writable from the
relevant hosts; understand whether the scheduler stages output at job end.

Choose the submission directory explicitly. `PBS_O_WORKDIR` records it; use
`cd -- "$PBS_O_WORKDIR" || exit 1` when that is the desired execution directory.
Quote shell paths and check inputs, software initialization, and scratch
visibility on compute nodes. Stage out required results before temporary storage
is reclaimed. Keep heavy payload execution on allocated compute resources.

For Bash scripts, run `bash -n job.pbs`; use the appropriate syntax check for
other interpreters. This does not validate scheduler directives. Preserve payload
exit codes through pipelines, cleanup, and stage-out. A final successful `echo`
must not hide a failed calculation. Validate application results explicitly when
downstream jobs require application-level success.

## Submit and record

Inspect the final script before an authorized submission. Example shell commands
use literal placeholders, not site defaults:

```bash
cd /absolute/submission/directory || exit 1
qsub -A PROJECT -q QUEUE job.pbs
```

Account/queue options can instead reside in the script where appropriate. Retain
the full returned job ID (including any suffix or array brackets), stdout,
stderr, command exit code, submission time/host/directory, script path/version,
and effective options in task notes or a project-appropriate log. No particular
receipt format or Python helper is required. Confirm the accepted job's settings
with `qstat -f JOB_ID`; successful submission only means the scheduler accepted it.

After verifying local support, dependency and array patterns include:

```bash
qsub -W depend=afterok:JOB_ID analysis.pbs
qsub -J START-END array.pbs
```

These fragments assume other required site settings are supplied by the script
or command. `afterok` checks the predecessor's exit status, not application
correctness. For arrays, use the implementation's documented index variable
(e.g. `PBS_ARRAY_INDEX` on implementations that provide it), map each index to
an input explicitly, and give tasks distinct output/scratch paths. Inspect
individual subjobs when diagnosing failures; a parent state is insufficient.

If submission times out, is interrupted, or returns no clear job ID, do not
blindly retry. Inspect the user's queue and candidate job details, matching
owner, name, submission time, working/output paths, and requested resources.
Consult retained history/accounting if supported, since a short job may already
have left the queue. A shared job name alone is not identity evidence. If the
outcome remains ambiguous, report the uncertainty and resolve it with the user
or site support before submitting another copy or cancelling a candidate.

## Inspect states, outputs, and exit codes

```bash
qstat -u "$USER"
qstat -f JOB_ID
qstat -Qf
```

Use the capability checks above before requesting JSON or finished-job history.
Avoid exposing `Variable_List` or full environment dumps; they can contain
credentials. Quote actual array IDs in shell commands to prevent glob expansion.

Common states include Q (queued), R (running), H (held), and E (exiting).
Finished states such as F or C, and other states, depend on implementation;
consult its definitions. Inspect scheduler comments, holds, dependencies,
requested/used resources, timestamps, output paths, and exit status to explain
a job's condition. Queued does not mean failed; exiting is not yet finished.
A missing job or failed scheduler query does not establish completion.

Read stdout/stderr and application outputs. Interpret nonzero or negative exit
codes using the local scheduler documentation and payload logs; do not assume
every code is a shell signal. Scheduler completion, even with exit status zero,
does not establish application-level correctness or scientific convergence.

## Modify or cancel identified jobs

```bash
qdel JOB_ID
qhold JOB_ID
qrls JOB_ID
qalter -l walltime=HH:MM:SS JOB_ID
```

Replace placeholders and check ownership/identity and requested scope first.
`qdel` can terminate a running job. `qhold` places a user hold; it is not a running
process suspension command. `qrls` releases a hold and may make the job runnable.
`qalter` changes only what the installation and job state permit. For arrays,
distinguish a whole-array action from an individual subjob action.

Capture the result and recheck with `qstat`. An accepted request may take time to
settle; a vanished record alone does not prove how the job ended. Do not broaden
a failed targeted command into bulk cancellation or escalate to forced deletion
without a justified, authorized scope.

## Recover and report

Preserve logs and checkpoints. Determine whether failure came from limits,
walltime, filesystem capacity, the environment, execution, or application
validation. Follow software-specific restart rules; the scheduler cannot choose
a valid application restart. Check that the earlier attempt is no longer active
before an authorized replacement, and record the relationship between attempts.
Stop repeated retries when the same unresolved failure recurs.

Report actual job IDs, directories, script/resource changes, observed states,
exit codes, relevant outputs, and unresolved issues. Distinguish observations
from inferences. Do not promise unattended monitoring unless it is configured.
