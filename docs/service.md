# Authenticated ChemGraph service

Install one distribution with `pip install "chemgraph[service]"`. The optional
`chemgraph.service` package integrates Connect at upstream commit `7232594`,
including its Hermes acceptance-recovery and delegation improvements. There is
no standalone Git dependency or production fixture mode. Ordinary ChemGraph
imports and commands do not import service extras.

The initial deployment uses one POSIX service instance and one serialized graph
worker. Linux Python 3.11–3.13 is the service CI matrix; core platform checks remain
unchanged. Hermes runs separately and needs no scientific Python environment.
Start with a fresh state directory. Keep old Connect databases and artifacts for
inspection; there is no checkpoint/state migration.

## Install and configure

Copy `examples/service/config.toml`, select permitted capabilities/calculators,
and set separate peer, operator and notifier token environment variables (at
least 32 characters each). Configure your existing ChemGraph model credentials
on the service host. Set `public_url` to an HTTPS origin for remote access.

```sh
chemgraph service --config examples/service/config.toml doctor
chemgraph service --config examples/service/config.toml serve
```

`/health/live` is public. `/health/ready` requires authentication and returns 503
until storage, checkpoints, recovery, and the worker are ready. Operator metrics
are at `/admin/metrics`; `doctor` checks installed versions and SQLite health.
The service uses explicit `approval_mode="review"`, disables skill discovery
and specialist delegation, and has no host shell capability.

All existing ASE calculators are integration-supported: EMT; MACE Polar, MP,
OFF and ANI-CC; FAIRChem/UMA; AIMNET2; TBLite; NWChem and ORCA. Availability still
requires their dependencies, installed model resources, configured executables,
and operator permission. Every existing driver is retained, including dipole,
vibrations, IR and thermochemistry. Unsupported properties return the calculator's
scientific failure; the service never substitutes a calculator or weights.
EMT is the hermetic software acceptance fixture, not the service's scientific
scope. Scientific validation of a research workflow remains separate.

For model calculators, map `calculator:model-name` (or `calculator:default`) to
provisioned files in `model_roots`. FAIRChem additionally requires the matching
atomic-reference file. The existing loaders consume these exact resources.
Executable bindings contain a trusted command plus its resource/executable files;
include all relevant launchers and configuration files. Tool arguments cannot
supply commands/profiles. NWChem/ORCA working directories stay in the conversation
workspace. No arbitrary executable profile is pickled. Provision dependencies
and resources before approval; missing resources produce a blocked proposal.

No environment-dependent default calculator is selected in strict execution.
Set `default_calculator` explicitly or have the researcher select a method.
Supported invocation limits are `max_steps` and `max_atoms`; remote scheduler
limits are specified in the fixed operator-authored HPC templates.

## Exact decisions

Submit and inspect tasks with `chemgraph service send`, `status`, `tasks`, and
`operations`. All remote commands accept `--url` and `--token-env`. Peer replies
can answer clarification interrupts, but cannot authorize calculations.

The operator's operations response includes the prepared action, resolved
arguments, immutable input checksums, resource identities, configuration identity,
limits, digest, and pending interrupt ID. The proposal executes the saved input
bytes even if the original file changes. Tampering with the saved snapshot or
changing a bound resource blocks invocation. Newly prepared inputs or settings
require another decision.

```sh
chemgraph service operations TASK_ID --token-env CHEMGRAPH_OPERATOR_TOKEN
chemgraph service approve OPERATION_ID --digest DIGEST --interrupt-id INTERRUPT_ID --expires-in 600 --token-env CHEMGRAPH_OPERATOR_TOKEN
chemgraph service reject OPERATION_ID --digest DIGEST --interrupt-id INTERRUPT_ID --message 'Revise this calculation' --token-env CHEMGRAPH_OPERATOR_TOKEN
```

Every action in a batch needs a decision; mixed approvals and rejections are
supported in checkpoint order. Expiry is checked again immediately before
execution. Blocked preparations cannot be approved. When an input is produced
by an earlier action in the same batch, the dependent call receives a rejection
explaining that it must be proposed in a later turn. This creates successive
reviews with actual input bytes.

## Recovery and finalization

Task, graph thread, tool operation, and remote job identifiers describe different
lifecycles. The journal records preflight, durable invocation intent, returned
result, and finalized artifacts separately. A preflight failure establishes
nonexecution. Intent without a saved return remains uncertain across restart.
A scientific failure returned by ASE is still a saved execution result.

Completed operations replay saved results. Saved returns retry artifact capture
without invoking a calculator again. Uncertain invocations and unfinished artifact
capture block subsequent native mutations in that workspace, including after
restart. Operators inspect evidence and use supported reconciliation:

```sh
chemgraph service reconcile OPERATION_ID --mode result --result-file recovered-result.json --evidence 'Describe the verified result evidence' --token-env CHEMGRAPH_OPERATOR_TOKEN
chemgraph service reconcile OPERATION_ID --mode job --batch-id EXISTING_JOB_ID --evidence 'Describe the gateway tracker evidence' --token-env CHEMGRAPH_OPERATOR_TOKEN
chemgraph service reconcile OPERATION_ID --mode nonexecution --evidence 'Describe evidence that invocation did not occur' --token-env CHEMGRAPH_OPERATOR_TOKEN
chemgraph service reconcile OPERATION_ID --mode finalize --evidence 'Artifact storage repaired' --token-env CHEMGRAPH_OPERATOR_TOKEN
chemgraph service reconcile OPERATION_ID --mode abandon --evidence 'Describe why tracking is being retired' --token-env CHEMGRAPH_OPERATOR_TOKEN
```

Recovered results must come from inspected scientific output, not guessed values.
Confirmed nonexecution creates a fresh pending decision before any retry.
Abandonment retires the conversation mapping and reports failure without claiming
successful science or confirmed cancellation. Remote cancellation remains
unconfirmed until the backend provides evidence; tracking continues otherwise.

Artifact registration also retains the saved graph outcome. Repair storage and
run `chemgraph service finalize-task TASK_ID --token-env CHEMGRAPH_OPERATOR_TOKEN`,
or restart, to retry registration independently. Artifact downloads require the
owning peer token; the workspace itself is never served. Immutable copies include
checksums. Notification retries and ambiguous delivery reconciliation are separate
from calculation execution.

## Coordinated backup and restore

1. Stop the service and notification helper, and verify both exited. Record the
   installed ChemGraph version, configuration, model-resource checksums and
   remote-job tracker locations. Stopping the worker is not scientific cancellation.
2. Copy the entire state directory as one unit: `service.db`, `a2a.db`,
   `checkpoints.db`, `sessions.db`, their WAL/SHM files if present, `profiles/`, and
   all `artifacts/` (including hidden prepared inputs and immutable snapshots).
   Also back up any separate remote execution trackers and operator resources.
3. Restore into a new directory on an isolated compatible host. Restore the same
   configuration, resource paths/checksums and secret references. Run `doctor`.
4. Start exactly one instance. Check `/health/ready`, `/admin/metrics`, task
   operations, artifact hashes and notification states before allowing traffic.
   Reconcile uncertain native/remote work; never resubmit based only on a lost
   acknowledgement. Test restoration with the hermetic service recovery suite.

Never independently restore just one database while leaving newer checkpoints
or artifacts alongside it. The stores deliberately keep separate responsibilities.

## Hermes and verification

Copy `integrations/hermes/` into the Hermes deployment. See its README for the
stdlib plugin and lightweight notification helper. Existing Telegram/WhatsApp
behavior is retained; Discord identity/delivery, attachments, campaigns and
distributed orchestration remain later roadmap milestones.

```sh
ruff check .
pytest tests/test_prepared_calculators.py
pytest tests/ -k "not tblite"
```

Service tests use authenticated in-process HTTP, scripted models, real EMT,
mocked other calculators and remote backends, and temporary journals/checkpoints.
Do not enable live LLM or HPC flags. The calculator contract module is run
separately so TBLite cases are not lost to the normal core filter.
