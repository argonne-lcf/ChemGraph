---
name: pbs-hpc
description: Prepare PBS HPC jobs, choose allocation and filesystem settings, submit and monitor jobs, and diagnose job output. Use for qsub, qstat, PBS scripts, allocations, and ChemGraph execution on Polaris, Aurora, or Crux.
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
Use attached HPC tools where available; otherwise prepare the script and state
where the user must submit it. Workspace workers must return chemistry
execution to their supervisor's chemistry specialist.

Read the applicable site reference before choosing launch settings:
[Polaris](references/polaris.md), [Aurora](references/aurora.md), or
[Crux](references/crux.md). Consult the linked official guides for current
queue limits and site policies.

## Prepare and track a job

1. Read [the PBS template](assets/job.pbs.template) and write a completed copy
   into the execution filesystem. Replace every placeholder, keep PBS directives
   before executable statements, and quote shell paths. Choose the launch command
   for the target site and application.
2. Check input visibility on compute nodes. Arrange staging first when the
   submitting host and workers do not share files. Validate the script syntax
   with `bash -n` when the execution environment provides Bash.
3. Submit once with `qsub job.pbs` on the authorized submission host, following
   existing tool approvals. Record the complete returned scheduler job ID.
4. Inspect with `qstat -f JOB_ID`; distinguish queued, held, running, and terminal
   states. Explain scheduler comments rather than submitting duplicate jobs.
5. Inspect stdout/stderr and application result files. A job disappearing from
   the active queue does not prove success. Use available job history and exit
   status, and report missing evidence explicitly. Cancel only the requested job.

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
