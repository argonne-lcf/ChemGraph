# Skill-guided HPC batch execution

Status: initial implementation complete; hermetic checks passed.
Live Polaris acceptance and user-supplied CUDA gRASPA remain deployment follow-ups.

## Architecture and compatibility

Implement scheduler-facing tools in `chemgraph.tools.hpc`, independent of
ExecutionBackend futures and existing MCP batch/task tracking. Use the existing
IRI module's authentication/resource resolution and the shared Globus manager.
Keep legacy IRI write gates, transfer layout defaults, imports, and MCP contracts.

Bind typed `[hpc.targets.<name>]` configuration to each agent's registry. Honor
selected CLI configuration and explicit tool restrictions, including empty
catalogs. No global target state or independent configuration rediscovery.

Expose target discovery, staging/retrieval, transfer status, bounded remote
inspection, submission, job status/listing, and cancellation. Explicitly apply
Deep Agent reviews to mutations and remote file inspection.

## Transfer and execution invariants

- Explicit collection path mappings preserve relative names and directories.
- Globus authentication refreshes tokens and never prompts inside a tool.
- Stage a checksummed local snapshot to a unique remote run directory.
- Changed inputs require a fresh run directory and new transfer.
- Require successful staging and unchanged identities before submitting.
- Serialize mutations to the same local run; snapshot/evidence files are durable.
- Record the complete request and target identity before exclusive submission marking.
- Accepted duplicate calls return the saved job ID; unknown outcomes never resubmit.
- Reconcile saved operations or unique matching scheduler records; absent,
  ambiguous, or incompletely searched records leave submission unknown.
- Restart uses saved target/resource identities, not current target defaults.
- Cancellation requests do not establish terminal scheduler state.
- Preserve scientific success separately from scheduler status.
- No global job index, daemon, SQLite job store, or shutdown-triggered cancellation.

The local guarantee applies to one canonical run directory on a POSIX host.
Cross-controller coordination and external modification of remote files are
outside this version. Reopening a run is supported; moving it is not.

## Skills and chemistry

Add hpc-batch while preserving direct qsub guidance in pbs-hpc. Select one
submission method per run. Diagnostics use separate directories from production.
Reuse site/environment references, existing ASE schemas and run_ase_core.

Use compute-visible or run-relative structure/model/output paths and set the
remote working directory and CHEMGRAPH_LOG_DIR consistently. Preserve scientific
parameters. Launch resources are passed through JobSpec, not #PBS comments.
The packaged ASE runner guards against login-node execution and retains exit
codes for success, nonconvergence and failure. No automatic software installation.
Existing SYCL/H2O gRASPA stays unchanged; CUDA integration follows separately.

## Delivery and acceptance

Keep reviewable implementation units for shared clients, target/staging models,
submission/recovery, and agent/skill integration. Run ruff and the full hermetic
suite excluding tblite. Regression coverage includes legacy behavior, authentication,
path mappings, failed transfers, input changes, parallel calls, crashes/lost
responses, reconciliation, history, cancellation, approvals and configuration.
Exercise packaged assets and an agent preparing, staging, submitting, and a new
agent monitoring; verify real ASE/EMT in a separate simulated compute directory.

The public ALCF guide and IRI reference models provide the implemented contract;
the deployed OpenAPI endpoint was inaccessible during development. Do not claim
live deployment validation from mocked contracts.

Opt-in live acceptance: diagnostic job, ASE/EMT, production calculator, and later
CUDA gRASPA. Retain transcript, compute hostname/environment evidence, full job ID,
scientific results and artifacts; verify monitoring after restarting ChemGraph.

Use the bundled `hpc-batch` skill for the staged submission and recovery workflow.

Verification: `ruff check .` passed; full `pytest tests/ -k "not tblite"`
passed with 1,762 passed, 21 skipped and 2 deselected. After the final manifest
integrity guard, the focused HPC suite passed all 49 tests. No live service
test flags were enabled.
