---
name: pbs-hpc
description: Prepare PBS HPC jobs, choose allocation and filesystem settings, submit and monitor jobs, and diagnose job output. Use for qsub, qstat, PBS scripts, allocations, and ChemGraph execution on Polaris, Aurora, or Crux; also Polaris and Aurora environments, MPI/GPU affinity, monitoring, and Aurora /soft access troubleshooting.
compatibility: Submission requires PBS commands on an authorized submission host; compute execution requires an allocation and the site's application environment.
license: Apache-2.0
metadata:
  authors: Murat Keceli
  maintainers: tdpham2
---

# PBS and HPC workflows

## Establish the environment

Identify the target system, submission host, project/account, queue, walltime,
node count, required filesystems, and application environment. Obtain missing
values from the user or deployment configuration. Do not copy another user's
allocation, endpoint, private path, or environment.

Check whether `execute` runs on that host. A local shell on a laptop cannot run
remote PBS commands merely because a skill or an HPC MCP server is attached.
When the shell runs on the submission host, write the job files and submit with
PBS commands directly. Otherwise use [iri-hpc](../iri-hpc/SKILL.md) for ALCF remote
submission, suitable attached HPC tools, or prepare the script and state where
the user must submit it. Preserve the user's selected execution method.

Read the applicable site-reference sections before choosing launch settings:
[Polaris](references/polaris.md), [Aurora](references/aurora.md), or
[Crux](references/crux.md). Consult the linked official guides for current
queue limits and site policies.

For a simple Polaris job, start with [queues](references/polaris.md#queues),
[filesystems](references/polaris.md#filesystems),
[application modules](references/polaris.md#software--modules), and
[network proxy setup](references/polaris.md#internet-proxy-login-and-compute-nodes).
For single-process ASE launch settings, use the
[ASE batch example](../chemgraph/references/ase-batch.md). Consult the site's MPI,
affinity, and MPS/MIG sections when the application needs those features.

For Aurora, start with [system basics](references/aurora.md#system-basics) for
filesystems and the application environment, and
[queues and submission](references/aurora.md#pbs-queues-and-submission). Consult
PALS, GPU affinity, monitoring, and `/soft` troubleshooting sections as needed.
Use the [Crux reference](references/crux.md) for its site-specific settings.

## Prepare and track a job

For ASE calculations, also read the `chemgraph` skill's
[Python batch example](../chemgraph/references/ase-batch.md). Write the calculation
script in the workspace and run it inside PBS; no MCP server or Parsl is needed.

1. Read [the PBS template](assets/job.pbs.template) and write a completed copy
   into the execution filesystem. Replace every placeholder, keep PBS directives
   before executable statements, and quote shell paths. Choose the launch command
   for the target site and application.
2. Check input visibility on compute nodes. Arrange staging first when the
   submitting host and workers do not share files. Validate the script syntax
   with `bash -n` when the execution environment provides Bash.
3. Submit once on the authorized submission host, following existing tool approvals.
   Save submission evidence and the complete scheduler ID in the run directory.
4. Inspect with `qstat -f JOB_ID`; distinguish queued, held, running, and terminal
   states. Explain scheduler comments rather than submitting duplicate jobs.
   For a submission-only request, report the accepted ID and observed status.
5. When results are requested, monitor to a terminal state and inspect stdout/stderr
   and application result files. A job disappearing from the active queue does not
   prove success. Use available job history and exit status, and report missing
   evidence explicitly. Cancel only the requested job.

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
