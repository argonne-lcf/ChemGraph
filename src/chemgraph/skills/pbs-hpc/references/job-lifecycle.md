# Submit, monitor and recover direct PBS jobs

Use only for authorized direct-qsub work on the submission host. For an IRI run,
follow `iri-hpc` instead. Complete input staging, inspect files and resolve all
placeholders before submission. Existing tool approvals still apply.

Run this submission sequence from the fresh host run directory after completing
and inspecting the files:

```bash
set -euo pipefail
bash -n job.pbs
test ! -e job.id
if ! (set -C; : > submission.started) 2>/dev/null; then
    echo "Submission already attempted; inspect PBS before retrying." >&2
    exit 1
fi
qsub job.pbs > job.id 2> qsub.stderr
test -s job.id
cat job.id
```

Keep the marker and stderr if `qsub` fails or returns no ID: acceptance can be
uncertain. Inspect PBS by job name and run directory before any retry. A later
agent session reads the existing `job.id`, checks that job with `qstat -f` (or
`qstat -xf` for retained history), and inspects output files without resubmitting.
An explicitly requested retry uses a fresh directory after resolving the prior
job's state. Cancellation uses `qdel JOB_ID` only when requested.

PBS command reference: [ALCF running jobs](https://docs.alcf.anl.gov/running-jobs/).

## ChemGraph execution boundaries

ChemGraph's `hpc_configs` Parsl configurations for these systems use
`LocalProvider` inside existing allocations; they do not acquire a PBS
allocation automatically. Aurora and Crux require `PBS_NODEFILE`; Polaris has
a local/testing fallback, which is not evidence of a valid allocation.

Use `CHEMGRAPH_WORKER_INIT` or the configured Python environment for worker
setup. Check the deployment's shared filesystem assumption: Globus Compute
workers may not see files written by the submitting server. A Globus endpoint
has its own execution configuration and is distinct from direct `qsub` usage.

Inspect `qstat -f JOB_ID` for queued, held, running and terminal states; explain
scheduler comments instead of resubmitting. Inspect stdout/stderr and application
results, using retained history when available. A job disappearing from the active
queue does not establish scientific success; report missing evidence explicitly.
