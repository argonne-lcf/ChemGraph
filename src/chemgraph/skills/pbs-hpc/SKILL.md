---
name: pbs-hpc
description: Prepare PBS HPC jobs, choose allocation and filesystem settings, submit and monitor jobs, and diagnose job output. Use for qsub, qstat, PBS scripts, allocations, and ChemGraph execution on Polaris, Aurora, Crux, or Bebop; also Polaris and Aurora environments, MPI/GPU affinity, monitoring, and Aurora /soft access troubleshooting.
compatibility: Submission requires PBS commands on an authorized submission host; compute execution requires an allocation and the site's application environment.
license: Apache-2.0
metadata:
  authors: Murat Keceli
  maintainers: tdpham2
---

# PBS and HPC workflows

## Establish the environment

Identify the target system, submission host, project/account, queue, and walltime.
Obtain missing values from the user or deployment configuration; Polaris ChemGraph
jobs use the [shared environment](references/polaris.md#chemgraph-environment) by
default. Do not copy another user's allocation, endpoint, private path, or environment.

Check whether `execute` runs on the submission host: a laptop shell cannot run remote
PBS commands because a skill or MCP server is attached. Preserve the user's selected
execution method. Without one, submit directly when the shell is on the submission
host; otherwise use [iri-hpc](../iri-hpc/SKILL.md) for ALCF or suitable attached HPC
tools. If neither is available, prepare the script and say where to submit it.

Site references: [Polaris](references/polaris.md), [Aurora](references/aurora.md),
[Crux](references/crux.md), [Bebop](references/bebop.md). Check the linked official
guides for current queue limits.

## Prepare and track a job

1. Make a fresh run directory on a filesystem compute nodes can see (on Polaris,
   under `/eagle` or `/grand`, not `/home`). Write `input.json` with absolute paths as
   in the [ASE batch example](../chemgraph/references/ase-batch.md) and copy the
   `chemgraph` skill's [calculate.py](../chemgraph/assets/calculate.py) unchanged.
2. Write `job.pbs`:
   - **Polaris ChemGraph calculation:** copy the
     [Polaris template](assets/polaris-chemgraph.pbs.template) and fill only
     `JOB_NAME`, `PROJECT`, `QUEUE` (usually `debug`), and `WALLTIME`.
   - **Other sites or applications:** fill the [generic template](assets/job.pbs.template)
     from the site reference; keep `#PBS` lines before commands and quote paths.
3. Check the files (`bash -n job.pbs`, parse `input.json`), then submit once with the
   sequence below from the run directory, following existing tool approvals.
4. Inspect with `qstat -f JOB_ID`: distinguish queued, held, running, and finished,
   and explain scheduler comments instead of resubmitting. For a submission-only
   request, report the ID and observed state.
5. When results are requested, monitor to a terminal state, then read the job output
   and result JSON. A job leaving the queue does not prove success; use `qstat -xf`
   history and report missing evidence. Cancel (`qdel JOB_ID`) only when asked.

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

If `qsub` fails or prints no ID, keep the marker and stderr: acceptance can be
uncertain, so inspect PBS by job name and run directory before any retry. A later
session reads `job.id` and checks that job without resubmitting. A requested retry
uses a fresh directory after resolving the prior job.

## ChemGraph execution boundaries

ChemGraph's `hpc_configs` Parsl configurations run inside an existing allocation
(`LocalProvider`); they never submit a PBS job. Globus Compute endpoints have their
own execution configuration and may not see files written by the submitting host.
