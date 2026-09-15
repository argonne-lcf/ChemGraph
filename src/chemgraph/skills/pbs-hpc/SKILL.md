---
name: pbs-hpc
description: Prepare PBS HPC jobs, choose allocation and filesystem settings, submit and monitor jobs, and diagnose job output. Use for qsub, qstat, PBS scripts, allocations, and ChemGraph execution on Polaris, Aurora, or Crux; also Polaris and Aurora environments, MPI/GPU affinity, monitoring, and Aurora /soft access troubleshooting.
compatibility: Submission requires PBS commands on an authorized submission host; compute execution requires an allocation and the site's application environment.
---

# PBS and HPC workflows

## Establish the environment

Identify the target system, submission host, project/account, queue, walltime,
node count, required filesystems, and application environment. Obtain missing
values from the user or deployment configuration. Do not copy another user's
allocation, endpoint, private path, or environment.

Check whether `execute` runs on that host. A local shell on a laptop cannot run
remote PBS commands merely because a skill or an HPC MCP server is attached.
When the shell cannot submit there, use attached HPC tools if they support the
requested workflow; otherwise prepare the script and state where to submit it.

Read the applicable site reference before choosing launch settings:
[Polaris](references/polaris.md), [Aurora](references/aurora.md), or
[Crux](references/crux.md). Consult the linked official guides for current
queue limits and site policies.

The Polaris reference includes login-node usage, modules, network proxy setup,
queues, MPI/OpenMP examples, CPU/GPU affinity, MPS/MIG, and storage. Read it for
Polaris environment setup as well as job submission.

For a complete chemistry workflow, read [Polaris ASE calculations](references/polaris-ase.md).
Prefer its direct PBS workflow for individual ASE jobs when the shell runs on
the submission host. It also covers MCP/Parsl for ensembles and worker reuse.
Preserve the user's explicit choice of execution method.

The Aurora reference covers hardware, queue selection, PALS, Intel GPU hierarchy
and affinity, monitoring, and group-restricted `/soft` access. Read it for Aurora
operations and environment troubleshooting as well as job submission.

## Prepare and track a job

1. For direct Polaris ASE jobs, use the templates and helper in the
   [ASE recipe](references/polaris-ase.md). For other applications, read
   [the PBS template](assets/job.pbs.template) and write a completed copy into
   the execution filesystem. Replace every placeholder, keep PBS directives
   before executable statements, and quote shell paths. Choose the launch command
   for the target site and application.
2. Check input visibility on compute nodes. Arrange staging first when the
   submitting host and workers do not share files. Validate the script syntax
   with `bash -n` when the execution environment provides Bash.
3. For direct Polaris ASE jobs, run `bash submit_ase.sh` from the staged run directory;
   it records the attempt and saves `job.id`. For other jobs, submit once with
   `qsub job.pbs`. Follow existing tool approvals and retain the full scheduler ID.
4. Inspect with `qstat -f JOB_ID`; distinguish queued, held, running, and terminal
   states. Explain scheduler comments rather than submitting duplicate jobs.
5. Inspect stdout/stderr and application result files. A job disappearing from
   the active queue does not prove success. Use available job history and exit
   status, and report missing evidence explicitly. Cancel only the requested job.
   For direct ASE jobs in a later agent session, read the run directory's `job.id` and
   inspect that same job and its results; do not rerun the submission helper.

PBS command reference: [ALCF running jobs](https://docs.alcf.anl.gov/running-jobs/).

## ChemGraph execution boundaries

Direct PBS jobs run the bundled ASE helper inside the allocation without an
MCP server or Parsl. For the optional Parsl route, the default
`allocation_mode="existing"` uses `LocalProvider` inside an existing
allocation. Aurora and Crux require `PBS_NODEFILE`; Polaris has a local/testing
fallback, which is not evidence of a valid allocation. Polaris also supports
explicit `allocation_mode="pbs"`: `PBSProProvider` acquires allocations from the
login node. In that mode, let Parsl submit allocations; do not also run `qsub`
for the same MCP calculation.

Use `CHEMGRAPH_WORKER_INIT` or the configured Python environment for worker
setup. Check the deployment's shared filesystem assumption: Globus Compute
workers may not see files written by the submitting server. A Globus endpoint
has its own execution configuration and is distinct from direct `qsub` usage.
