# Integrate Connect with all ChemGraph ASE calculators

Status: implemented in two stacked branches; cross-platform CI pending. Updated: 2026-10-13.

Use two stacked PRs. First revise [#258](https://github.com/argonne-lcf/ChemGraph/pull/258)
on `feature/scoped-tool-runtime`, preserving its isolation fixes, with prepared
ASE actions and recoverable scoped execution. Then integrate current Connect as
optional `chemgraph.service` on a branch based on that PR. Append focused commits;
do not rewrite the shared branch.

## PR 1: prepared ASE actions and recoverable scoped execution

A versioned serializable preparation binds operation, tool, resolved arguments,
input checksums, concrete calculator resources, configuration identity, and limits.
Reuse ASEInputSchema and calculator schemas. Freeze referenced input bytes before
review, execute that preparation, reject tampering, and revalidate resources.
Changed settings or newly prepared input bytes require a new digest and decision.
An omitted calculator requires an explicit operator default or clarification.
Dependent actions whose input files do not yet exist require successive reviews.

Support EMT; MACE Polar, MP, OFF, and ANI-CC; FAIRChem/UMA; AIMNET2; TBLite;
NWChem; and ORCA. Preserve all drivers and calculator/driver combinations,
scientific settings, aliases, charge/spin, devices, precision, model selection,
and executable bindings. Missing resources must be provisioned before approval.
Use serializable trusted executable descriptors, never arbitrary pickled profiles.
Availability and optional imports remain lazy; never substitute a method/model.
EMT is a hermetic test fixture, not a product restriction.

Execution contract version 2 separates preflight, invocation, and finalization.
Persist invocation intent before calling the tool, and returned results before
artifact capture. Distinguish confirmed nonexecution, uncertain invocation,
results awaiting finalization, and finalized execution. Save scientific failures.
Replay saved results and retry finalization without another calculation. Persist
the output baseline and block workspace mutations until outputs are immutable,
including after restart. Supply a disk-backed test controller; production journals,
authentication, transport, and operator APIs belong to PR 2. Reject v1 controllers.

Optional strict policy enforces configured capabilities and calculators at every
native execution boundary, including attached tools and delegated calls. Separate
framework helpers from science. Unsupported non-ASE preparation fails before
invocation. Preserve unscoped behavior and existing workspace/document isolation.

Commit groups: plan/API documentation; preparation/calculators; staged execution
and recovery coverage.

## PR 2: integrated service

Import current Connect including Hermes recovery improvements. Add `[service]`
and lazy `chemgraph service` dispatch; remove production fixture mode and Git pins.
Implement v2 control, exact approvals, expiry, rejection and mixed decisions,
configurable capabilities, and authenticated native/remote reconciliation.
Recover returned results or existing jobs, confirm nonexecution before a fresh
decision, or abandon without claiming success or confirmed cancellation.

Reuse MainAgentSession, graph checkpoints, readable history, and execution backends.
Keep task, operation, graph thread, and remote job identities distinct. Preserve
authenticated immutable artifacts and the notification outbox; retry finalization
and delivery independently. Hermes and its notification helper remain independently
deployable. Update roadmap ownership, install/health checks, and coordinated
backup/restore instructions.

## Verification

Parameterized mocked tests cover every calculator and MACE variant, field/alias
preservation, concrete resources/executable bindings, and every existing driver.
Use real EMT locally, mock other engines, and never download models or run live
LLMs/HPC jobs. Run the calculator contract suite separately so `not tblite` does
not suppress its TBLite cases. Cover tampering, changed inputs/resources,
capability rejection, preflight failures, interrupted invocation, persistence and
artifact failures, restart, replay, and workspace protection across graph paths.

Run `ruff check .` and `pytest tests/ -k "not tblite"` with isolated session storage.
PR 2 adds authenticated recovery flows, installations with/without service extras,
Linux Python 3.11–3.13 checks and existing core platform checks.

## Boundaries

Fresh service state; preserve old Connect state for inspection without migration.
Initially one instance and one serialized graph worker. Calculator integration is
separate from scientific validation of a research workflow. Discord identity and
delivery, attachments, campaigns, and distributed orchestration remain later work.
Budget approximately 8,000–10,000 net lines across both PRs, refined from actual diffs.

Current diff size: approximately 8,000 net lines including PR #258's original
scoped runtime, the prepared execution additions, integrated service, tests and
deployment documentation.
