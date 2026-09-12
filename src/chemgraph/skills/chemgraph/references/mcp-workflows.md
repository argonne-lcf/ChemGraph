# Chemistry through attached MCP tools

Read the actual attached schemas; tool availability depends on server and
configuration. The ASE HPC server exposes `run_ase_single`, `run_ase_ensemble`,
and result-reading tools. Job tools include `check_job_status`,
`get_job_results`, `list_jobs`, and `cancel_job`.

1. Identify input visibility before submission. Local input paths are local to
   the MCP server, not automatically local to the agent. The ASE server can
   embed structures found on its submitting host for remote workers; pre-staged
   remote paths instead refer to the worker filesystem.
2. When Globus Transfer tools are attached, use `transfer_files` according to its
   schema and inspect `check_transfer_status` until staging succeeds. On failure,
   report the transfer failure without starting calculations.
3. Use the returned compute-visible directory for an ensemble's
   `remote_structure_directory`. Collection-relative listing paths and
   worker-visible paths may differ. Do not prepend the agent's virtual mount to
   a remote path or fabricate an endpoint identifier. Destination selection must
   follow the attached schema; do not assume a `compute_system` argument exists.
4. Preserve calculator/model/driver/device choices. Honor the selected tool's
   schema for each parameter; do not infer CUDA solely from an HPC hostname.
5. If the result is `submitted`, retain its `batch_id`. Use status/result tools
   to distinguish pending, failed, and completed calculations. Use a wait tool
   only when one is actually attached. Avoid indefinite tight polling.
6. Report returned success/failure counts and scientific results. Include batch
   IDs for pending work and the location of full results. A successful transfer
   or submission does not establish successful calculation completion.

For unavailable capabilities, describe the missing tool or setup requirement.
Do not generate an alternative unapproved shell simulation to bypass missing
chemistry tools or the workspace worker's role.
