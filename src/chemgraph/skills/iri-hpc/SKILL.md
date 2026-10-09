---
name: iri-hpc
description: Stage files with Globus Transfer and submit, monitor, and retrieve remote PBS jobs through configured IRI HPC tools, including from a laptop without SSH or qsub.
license: Apache-2.0
compatibility: Requires a POSIX agent host, configured IRI and Globus resources, and an existing compute environment.
metadata:
  authors: Thang Pham
  maintainers: tdpham2
---

# Remote jobs through IRI

Use this workflow when the user selects IRI remote batch execution. Read the
`pbs-hpc` site reference for environment requirements, but use `hpc_*` native
tools for this workflow. Direct qsub remains appropriate on an authorized PBS
submission host when the user selects it. Never mix submission methods for a run.

1. Load `hpc_list_targets` and inspect configuration. Missing targets require
   configuration, not guessed projects, collections, paths, or environments.
   Authenticate Transfer in a terminal with
   `python -m chemgraph.execution.globus_transfer`. Add repeatable
   `--collection <collection-id>` options for managed collections requiring
   data_access consent. IRI uses separate Facility API credentials from
   `alcf_auth` (`ALCF_API_TOKEN` or its supported token cache); Transfer login does not
   authenticate IRI. Never put tokens or authorization codes into tool arguments.
2. Create a fresh absolute host run directory beneath the target's local_root.
   Native tools run on the agent host; `/workspace` may be a virtual file-tool
   path. Obtain the application command, working directory, required files and
   software environment, launch requirements, expected outputs, and exit-code
   meanings from the application skill. Prepare its inputs with run-relative
   or known compute-visible paths, never laptop absolute paths.
3. Adapt [launch.sh.template](assets/launch.sh.template). Current IRI targets use
   PBS: copy [pbs-launch.sh](../pbs-hpc/assets/pbs-launch.sh) alongside the launch
   script and include it in staging. It validates the allocation before executing
   the application; invoke it with `bash` because staging does not preserve
   executable permissions. Fill `ENVIRONMENT_SETUP` with the required setup and
   `APPLICATION_LAUNCH_COMMAND` with an executable plus shell-quoted arguments,
   without a leading `exec`. Put compound shell commands in a separate staged
   script invoked with `bash`. Request `arguments` are forwarded unchanged.
   Validate script syntax without running the application on a login node.
   Explicitly select resources in the submission request: #PBS comments are not
   scheduler directives here.
4. If needed, run a small separate diagnostic job first. Record compute hostname,
   Python/binary locations, modules, requested libraries and device visibility.
   Reuse verified environment setup; do not automatically install software.
   Diagnostic and production jobs always use separate run directories.
5. Load `hpc_transfer_files` and stage explicitly selected relative files with
   direction=stage and the configured target. The tool freezes a local snapshot
   and returns a unique remote_directory and transfer_id. Changed inputs require
   a fresh run and another transfer. Do not edit or transfer into the frozen run.
   A failure with phase=prepare and retry_safe=true leaves no staging snapshot
   or manifest: correct authentication or payload errors and retry the same
   directory. Preparation refreshes credentials and builds the payload; it does
   not guarantee collection permissions or consent. Once submission begins,
   phase=submit and retry_safe=false mean the outcome may be unknown. Preserve
   evidence and inspect Globus task history; never automatically repeat a transfer
   or reset its manifest. Existing unknown attempts remain guarded, including
   those without a transfer_id (reported as transfer_unknown by status).
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
   Retrieval preparation failures reserve no paths and can be corrected and
   retried. Recorded retrieval attempts retain their reservations; do not use
   overwrite to bypass an uncertain submission.
9. Return scheduler evidence, exit status, stdout/stderr, and artifact locations
   to the application workflow for interpretation. Scheduler completion alone
   does not establish application success; early failures may leave only stderr.
   Report uncertainty and missing outputs.

Loading tools replaces the active selection; load all tools needed in the next
step together. Keep this workflow in this agent. Cancel only a user-requested
job using hpc_cancel_job with its exact full job ID and run directory, then
monitor until scheduler evidence confirms the outcome.

Existing MCP batch IDs and Globus Compute task IDs are not PBS job IDs.
Credentials belong in authentication caches, not scripts, arguments or manifests.
No daemon is needed: accepted jobs survive agent exit. Cross-controller and
external remote-file modifications are outside the local evidence guarantee.
