---
name: pbs-hpc
description: Prepare, submit and inspect PBS jobs on Polaris, Aurora or Crux; resolve site environment and scheduler problems.
compatibility: Submission requires PBS commands on an authorized submission host; compute execution requires an allocation and the site's application environment.
license: Apache-2.0
metadata:
  authors: Murat Keceli
  maintainers: tdpham2
---

# PBS and HPC workflows

Establish the target system, submission host, project, queue, walltime, nodes,
filesystems and application environment. Use user/deployment values; ask for
missing choices. A laptop shell cannot run remote PBS commands merely because
an HPC tool is attached. Skills do not override existing action approvals.

## Prepare files

- Read the selected site's guide: [Polaris](references/polaris.md),
  [Aurora](references/aurora.md) or [Crux](references/crux.md). Open known paths
  directly; batch independent reads. Follow advanced links only for relevant
  launch or troubleshooting needs, or when explicitly requested.
- For ASE, use [the complete batch example](../chemgraph/references/ase-batch.md).
  For other applications, adapt [the PBS template](assets/job.pbs.template).
  Keep directives before executable commands and use quoted host paths.
- Verify compute-visible input/environment paths; stage inputs when filesystems
  differ. Validate with `bash -n` and application-specific input/syntax checks,
  without running the calculation on a login node. Resolve placeholders before
  submission; prepare-only requests may retain explicitly requested placeholders.
- For **prepare only**, report files and validation, then stop. Result files do
  not exist until the calculation runs. No submission or lifecycle read is needed.

## Submit or inspect

For direct PBS submission, monitoring or recovery, read
[the job lifecycle](references/job-lifecycle.md). Submit once on the authorized
host, preserve the submission marker, stderr and full job ID, and reconcile an
uncertain outcome before retrying. Scheduler completion is not scientific success;
inspect application results. Cancel only the requested job.

When the user selects IRI and configured `hpc_*` tools are available, read
`hpc-batch` instead. Use its staging/evidence workflow and typed scheduler
resources; do not apply the direct-qsub sequence to that run.

ChemGraph Parsl configurations use existing allocations, not automatic PBS
submission. Globus Compute and direct PBS have distinct execution/filesystem
settings; see the lifecycle reference when configuring these backends.
