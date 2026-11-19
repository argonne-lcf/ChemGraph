---
name: pbs-hpc
description: Prepare, submit, and monitor ChemGraph calculations through PBS on Polaris, Aurora, Crux, or Bebop. Use for ChemGraph qsub scripts, allocation and filesystem settings, qstat, and calculation results.
compatibility: Submission requires PBS commands on an authorized submission host; compute execution requires an allocation and the site's application environment.
license: Apache-2.0
metadata:
  authors: Murat Keceli
  maintainers: tdpham2
---

# PBS and HPC workflows

## Establish the environment

Use the user's selected submission method. For Polaris, Crux, or Aurora, if
the route is undecided, run `command -v qsub` in the execution shell. Use
direct PBS if it succeeds and [IRI](../iri-hpc/SKILL.md) if it fails.
For other PBS sites without `qsub`, use a suitable attached HPC tool or prepare
the script and say where to submit it.

Identify the target system, submission host, project/account, queue, and walltime.
Obtain missing values from the user or deployment configuration. On Polaris, Crux, and
Aurora, use the shared environment in the site reference by default.

Site references: [Polaris](references/polaris.md), [Aurora](references/aurora.md),
[Crux](references/crux.md), [Bebop](references/bebop.md). Check the linked official
guides for current queue limits.

## Prepare and track a job

1. Make a fresh run directory on a filesystem compute nodes can see (on Polaris,
   under `/eagle` or `/grand`, not `/home`). Write `input.json` with absolute paths as
   in the [ASE batch example](../chemgraph/references/ase-batch.md) and copy the
   `chemgraph` skill's [calculate.py](../chemgraph/assets/calculate.py) unchanged.
2. Write `job.pbs`:
   - **ChemGraph on ALCF:** copy the [Polaris](assets/polaris-chemgraph.pbs.template),
     [Crux](assets/crux-chemgraph.pbs.template), or [Aurora](assets/aurora-chemgraph.pbs.template)
     template. Fill `JOB_NAME`, `PROJECT`, `QUEUE` (usually `debug`), and `WALLTIME`.
   - **Other PBS sites:** fill the [generic template](assets/job.pbs.template)
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
