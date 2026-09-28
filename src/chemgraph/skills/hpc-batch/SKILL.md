---
name: hpc-batch
description: Stage files with Globus Transfer and submit, monitor, and retrieve remote PBS jobs through configured IRI HPC tools, including from a laptop without SSH or qsub.
license: Apache-2.0
compatibility: Requires a POSIX agent host, configured IRI and Globus resources, and an existing compute environment.
metadata:
  authors: ChemGraph contributors
  maintainers: tdpham2
---

# Remote HPC batch jobs

Use this workflow when the user selects IRI remote batch execution. Read the
`pbs-hpc` site reference for environment requirements, but use `hpc_*` native
tools for this workflow. Direct qsub remains appropriate on an authorized PBS
submission host when the user selects it. Never mix submission methods for a run.

1. Load `hpc_list_targets` and inspect configuration. Missing targets require
   configuration, not guessed projects, collections, paths, or environments.
2. Create a fresh absolute host run directory beneath the target's local_root.
   Native tools run on the agent host; `/workspace` may be a virtual file-tool
   path. Prepare all inputs and a launch script there. Preserve scientific
   settings and selected compute methods. Use relative paths inside input.json
   or known compute-visible paths, never laptop absolute paths.
3. Read the ASE example in [the chemistry reference](../chemgraph/references/ase-batch.md).
   For remote ASE use [calculate.py](assets/calculate.py) and
   [launch.sh.template](assets/launch.sh.template). Replace placeholders,
   supply relative structure/output/model paths, and validate script syntax
   without running a calculation on the login node. Explicitly select resources
   in the submission request: #PBS comments are not scheduler directives here.
4. If needed, run a small separate diagnostic job first. Record compute hostname,
   Python/binary locations, modules, requested libraries and device visibility.
   Reuse verified environment setup; do not automatically install software.
   Diagnostic and production jobs always use separate run directories.
5. Load `hpc_transfer_files` and stage explicitly selected relative files with
   direction=stage and the configured target. The tool freezes a local snapshot
   and returns a unique remote_directory and transfer_id. Changed inputs require
   a fresh run and another transfer. Do not edit or transfer into the frozen run.
6. Load `hpc_transfer_status` and require SUCCEEDED. Then load `hpc_submit_job`,
   review the request, and submit once. All approvals remain active. A failed or
   unknown response is not permission to retry in another directory.
7. Load `hpc_job_status` to monitor or reconcile the run. On restart, reopen the
   same directory; its saved target is authoritative. Unknown acceptance stays
   unknown unless scheduler evidence identifies the job unambiguously. Never
   delete submission.started or substitute another submission method.
8. Load `hpc_list_files` and `hpc_read_file` for bounded inspection. Filesystem
   operations may return task_id: poll it using operation_id. Report permission
   failures; use Globus retrieval if IRI filesystem inspection is unavailable.
   Use hpc_transfer_files direction=retrieve for selected artifacts, then poll
   the returned transfer_id. Retrieved files go under run_dir/retrieved.
9. Inspect scientific result fields, stderr and exit status. PBS completion alone
   is not scientific success. ASE exits 0 for convergence, 2 for nonconvergence,
   and 1 for failure; early failures may leave no result JSON. Report uncertainty.

Loading tools replaces the active selection; load all tools needed in the next
step together. Keep this workflow in this agent. Cancel only a user-requested
job using hpc_cancel_job with its exact full job ID and run directory, then
monitor until scheduler evidence confirms the outcome.

Existing MCP batch IDs and Globus Compute task IDs are not PBS job IDs.
Credentials belong in authentication caches, not scripts, arguments or manifests.
No daemon is needed: accepted jobs survive agent exit. Cross-controller and
external remote-file modifications are outside the local evidence guarantee.
CUDA gRASPA integration is not included; do not substitute the SYCL/H2O runner.
