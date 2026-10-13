# ChemGraph service roadmap

Build a private lab service for one or two researchers who interact through
Discord, approve scientific work, receive inspectable results, and eventually
delegate ongoing campaigns. Start with a small set of supported chemistry
workflows. Milestones 1 and 2 constitute the first usable release.

ChemGraph owns scientific planning, calculations, validation, and interpretation.
`chemgraph.service` owns task persistence, access control, authorization records, and delivery
tracking. Hermes provides the conversational interface. Calculation infrastructure
remains ChemGraph's responsibility; each supported workflow must include a tested
execution and artifact-collection path.

Establish trusted identities, exact-action approval, scientific provenance, and a
compatible runtime in Milestone 1. Milestone 2 hardens those foundations for daily
use. Prove a bounded scientific study before expanding project-management features.

## 1 Complete a scientific task through Discord

Support structure preparation, a supported geometry optimization or energy
calculation, and interpretation of the resulting outputs.

Before enabling the first research workflow, ChemGraph must publish a versioned
workflow definition covering supported structure formats and chemical domain,
charge and spin handling, the installed calculator/model and resolved settings,
input-size and execution limits, and scientific validation criteria. Select and
validate one research method with the lab; an installed calculator alone does not
establish suitability. This selection is a release prerequisite, not a decision
left to the model during execution.

Use a bounded local calculation for the first integration demonstration. Enable
only the approved workflow's tools and methods in the service, while retaining
ChemGraph's broader catalog for other applications. Research requests must use
the validated method explicitly; do not inherit environment-dependent calculator
defaults. If the research workflow requires remote computation, job tracking and
retrieval of the actual output files are part of its acceptance criteria.

Install one version of `chemgraph[service]` and test it with the deployed Hermes
version before connecting Discord. The two stacked PRs in the
[integration plan](chemgraph-connect-integration.md) establish prepared ASE actions
and the integrated service. Follow the [service guide](../service.md) for health
checks and coordinated backup/restore. Use the asynchronous session API with
review enabled so unattended service operation can pause for human decisions.

The user can supply SMILES or a supported structure attachment, answer
clarification questions, review the proposed method, approve execution, check
progress, and receive results. Unsupported requests receive a clear explanation;
ChemGraph must not silently substitute a different scientific method.

Bind each allowlisted Discord user to a distinct Connect principal using trusted
gateway identity. Persist the conversation-to-context mapping and recorded DM
destination. Derive stable submission identifiers from incoming Discord events;
persist distinct action identifiers if one event creates multiple submissions.
Replaying an event must reuse its identifiers. Model-supplied owner names, channel
IDs, and remembered task IDs cannot establish identity or authorization.

Implement authenticated Discord approve and reject commands in this milestone.
Verify that the responding human may decide the specific task. Bind each approval
to the exact proposal, resolved method/settings, immutable input IDs and checksums,
execution limits, and compatible workflow version. Changed inputs or methods
require a new proposal. Ordinary chat text cannot grant execution permission.

Add trusted inbound attachment ingestion: validate format, content, and size;
preserve the original bytes as an owner-scoped input artifact; then supply that
artifact to the scientific workspace. Record prepared structures separately so
users can trace transformations back to their supplied input.

Deliver a concise scientific explanation and a results package assembled from
execution records and finalized artifacts, containing:

- Input and final structures.
- Resolved method, model identity, settings, software versions, and units.
- Numerical results, convergence information, and scientific validation outcomes.
- Relevant calculation outputs and a README explaining the files.
- A machine-readable manifest linking the proposal, input artifacts, execution
  records, and output artifacts, with filenames and checksums.

Create this provenance from the first task; project organization is added later.
Keep execution completion, scientific validity, and delivery status distinct.
An optimization that finished without convergence must retain its outputs and
report that limitation rather than support a claim of a validated result.

Add a trusted delivery helper that retrieves registered artifacts from Connect and
uploads them through Hermes. Bind delivery to the authenticated task owner and
conversation. Construct attachment references from validated artifact records;
model-generated paths cannot authorize file access.

Use small native attachments for the initial workflows. Preserve the original
artifacts in Connect. Record delivery separately from scientific completion so a
failed upload never triggers another calculation.

Use EMT for integration demonstrations and a validated, domain-appropriate method
for research examples.

**Complete when:** a user requests and approves a calculation entirely through
Discord, downloads its results, and inspects the structure, numerical output, and
method without accessing the service machine. The resolved method and input bytes
match the approved proposal, and unsupported work is rejected before execution.

## 2 Make the service dependable for two users

Exercise the identity, workspace, conversation, and approval boundaries with both
users. Provide durable task status, cancellation requests, and delayed Discord
completion notifications. Keep a single serialized graph worker initially; make
queued work visible and use tracked asynchronous jobs for long calculations.

Fix the execution-boundary and recovery gaps identified in PR #258 and Connect.
Distinguish confirmed failures, completed operations, and operations whose outcome
is uncertain. Only explicit execution evidence can establish that a failed action
did not run. Never automatically repeat an uncertain external submission.

Cover failures before execution, crashes after execution but before recording the
result, restarts while awaiting approval, expired approvals, and interrupted native
operations. Add supported reconciliation actions for native operations as well as
remote submissions, recording the decision and evidence. Preserve completed results
when reporting or artifact delivery fails. Cancellation status must distinguish
confirmed cancellation from an unconfirmed request, and continue tracking work
whose stopping has not been confirmed.

Add automatic restart, health checks, useful operational logs, and a tested backup
and restore procedure covering task records, checkpoints, artifacts, execution
trackers, and conversation mappings. Test the pinned component versions together
before upgrades and preserve a compatible recovery path for existing tasks.

Persist attachment delivery receipts. Retry known delivery failures; reconcile
ambiguous outcomes. Allow users to request an existing result again without
repeating its calculation.

**Complete when:** both users work independently, restarts preserve tasks and
approval records, and replayed requests do not repeat completed calculations.
Routine restarts recover automatically; uncertain outcomes use a documented human
reconciliation flow without editing databases or reconstructing task history.

## 3 Delegate a bounded scientific study

Let ChemGraph propose a multi-step study with an objective, inputs, methods,
limits, and completion criteria. For example:

> Prepare these conformers, optimize them using the agreed method, compare the
> converged results, and produce a report.

A human approves the study through Discord. ChemGraph performs authorized
intermediate steps, validates outputs, handles partial failures, and reports
progress. Changes to scope, methods, or limits require renewed approval. Routine
steps inside the approved scope proceed without repeatedly interrupting the user.

Implement study authorization as a durable, bounded grant covering the approved
inputs, methods, permitted actions, and calculation/resource limits. Check the
grant before every consequential action and persist remaining authorization across
restarts. Per-tool approval records are the starting point; global approval bypass
does not implement a study grant.

Start with one small conformer study using the validated method and result
manifests from Milestones 1 and 2. Verify composition and settings are comparable
and exclude nonconverged results from validated energy rankings. Keep the existing
session and execution interfaces; distributed Academy orchestration is not a
prerequisite for this milestone.

**Complete when:** ChemGraph completes a small study and produces an
evidence-backed report, asking for human decisions at meaningful boundaries.
Attempts to exceed scope or budget pause for renewed authorization, including
after restart.

## 4 Organize work into scientific projects

Group structures, documents, calculations, reports, and conversations into
projects. Build artifact naming, search, reuse, and explicit sharing on the
owner-scoped uploads, downloads, and provenance established in the first release.

Resolve requests such as "compare this with yesterday's calculation" to specific
evidence. Keep remembered scientific claims traceable to their calculations or
sources, and reuse verified prior results when appropriate. Sharing must be
explicit; project membership or chat context must not silently broaden access.

Add expiring download links for outputs exceeding Discord's attachment limits.
These links need browser-compatible access; simply posting an API URL that
requires a bearer token will not suffice.

**Complete when:** users can revisit a study, find its evidence, reuse inputs, and
produce defensible comparisons.

## 5 Continue approved campaigns over time

Persist campaign objectives, findings, remaining authorization, pending decisions,
and next actions.

Resume bounded work after calculation completion, new data, a scheduled time, or
human approval. Provide pause and resume controls, stopping conditions, and
protection against overlapping iterations or catch-up floods after downtime.

ChemGraph can recommend a new direction, but expanding the campaign requires
renewed authorization. Enforce the approved tool set and compute budget throughout
the campaign.

**Complete when:** a campaign progresses across several days and restarts while
respecting its approved scope and limits.

## 6 Expand according to demonstrated demand

Add scientific workflows with validated examples, explicit supported scope,
reproducible outputs, and meaningful failure handling. Select capabilities from
actual user needs, such as thermochemistry, adsorption, docking, spectroscopy, or
literature-supported analysis.

Add Telegram and other clients through the same task, approval, and artifact
interfaces. External agents can submit work and retrieve results without
depending on a particular chat session.

**Complete when:** additional capabilities and clients reuse the established
scientific and service interfaces.

## Discord artifact delivery

Discord supports uploading files through its bot API. Hermes documents native
attachment delivery using `MEDIA:<local_path>` in a message. Verify that behavior
against the Hermes version pinned by Connect before relying on it in the service.
See the [Discord upload API](https://docs.discord.com/developers/reference#uploading-files)
and [Hermes CLI documentation](https://hermes-agent.nousresearch.com/docs/reference/cli-commands#sending-images-and-other-media).

Use this delivery sequence:

1. ChemGraph creates the scientific outputs and identifies the results to deliver.
2. Connect preserves finalized artifacts with their task ownership, IDs, filenames,
   sizes, and checksums.
3. A trusted delivery helper downloads the selected artifacts from Connect into a
   managed cache on the Hermes machine, using authorization scoped to the task.
4. Hermes uploads the file bytes to the user's recorded Discord destination and
   sends a short explanation of the results.
5. Connect records the delivery receipt separately from the calculation outcome.

ChemGraph and Hermes can run on different machines; authenticated artifact
downloads remove the need for a shared filesystem. A pathname on the ChemGraph
machine must be resolved through the artifact service before Hermes can upload
the file.

For example, after a results archive has been downloaded onto the Hermes machine:

```bash
hermes send --to "discord:<channel-id>" \
  "ChemGraph results are ready. MEDIA:/managed/cache/task-123/results.zip"
```

Replace the destination with the authenticated conversation's recorded channel
ID. Construct the local path in trusted delivery code. Hermes's `--file` option
reads a text message body; `MEDIA:` requests a native attachment, as described in
the [Hermes CLI documentation](https://hermes-agent.nousresearch.com/docs/reference/cli-commands).

The inspected Connect integration returns artifact references, and its delayed
notification helper sends text to configured Telegram recipients. Implementing
Discord delivery requires artifact retrieval and attachment upload, Discord
recipient routing, and support for Discord identities in the approval flow. The
existing authenticated artifact endpoint is the starting point for retrieval.
Connect's inspected submission endpoint accepts text only, so inbound attachments
also require an ingestion interface. Remote-result JSON may contain worker-local
paths; those files must be collected and registered before they are deliverable.

Keep the service's stored artifacts as the authoritative copies. Deliver a compact
report and results package automatically; make additional outputs retrievable on
request. For the initial release, report oversized-file limitations clearly and
retain the files. Add browser-compatible expiring download links in Milestone 4.

## Ownership and interfaces

| Area | Owner |
| --- | --- |
| Scientific objectives, methods, and workflow planning | ChemGraph |
| Calculations, execution, and scientific validation | ChemGraph |
| Scientific interpretation and reporting | ChemGraph |
| Task persistence, access control, approval records, and delivery tracking | `chemgraph.service` |
| Supported scientific workflow definitions and validation criteria | ChemGraph |
| Trusted chat identity, conversation mapping, and delivery integration | Independently deployed Hermes integration |
| Discord and Telegram interaction and message transport | Hermes |
| Approval of scope and consequential changes | Human |

ChemGraph proposes scientific actions. `chemgraph.service` records and enforces authorization.
Hermes presents decisions and returns authenticated human responses. Headless
operation can pause for approval without requiring an interactive terminal.

Retain A2A for programmatic access. Extend artifact handling with owner-scoped
retrieval, inbound structure uploads, attachment delivery records, and later
expiring download links. Keep approval records bound to the exact scientific
proposal and immutable inputs. Keep capability restrictions and study budgets
enforceable at execution time, independently of the model's proposed actions.

## Verification and release criteria

Before the first usable release, exercise at least ten representative requests
across both users, including clarification, approval, rejection, calculation
failure, restart, duplicate submission, and result retrieval.

Automate regression cases for:

- Both users' task, workspace, input, and output isolation; wrong-owner approvals
  and downloads; and forged identity or destination values.
- Replayed Discord events, lost submission acknowledgements, and conversation
  continuation after a gateway restart, without duplicate calculation execution.
- Approval rejection and expiry; altered inputs, settings, or workflow versions;
  and refusal to execute unsupported methods or exceed configured limits.
- Failure before execution, execution with a lost result acknowledgement, restart
  at approval/execution/checkpoint boundaries, native-operation reconciliation,
  and cancellation whose outcome is not yet confirmed.
- Malformed or oversized inbound structures, missing remote output files, invalid
  numerical results, nonconvergence, and a complete manifest for retained outputs.

Verify that downloaded Discord attachments match the stored artifacts by checksum.
Test wrong-owner access, failed uploads, ambiguous delivery, unsupported inputs,
and oversized-file handling. Confirm that delivery failures and repeated requests
for existing results do not rerun calculations.

Evaluate scientific validity separately from successful software execution.
Preserve inputs, methods, settings, versions, and outputs. Check convergence and
domain-specific validity before drawing conclusions. Maintain representative
scientific tasks with expected checks and outcomes alongside software tests.

Use hermetic software tests with scripted models and local fixtures, including EMT
where appropriate. Separately run live deployment acceptance with the pinned
Discord/Hermes integration and a validated research example. Record component
versions and acceptance evidence; passing fixture tests alone does not establish
scientific validity or live delivery.

Judge the first release by whether both users can complete ordinary supported
scientific work without developer intervention. Exceptional uncertain outcomes
may require a human decision through the supported reconciliation flow.

## Defaults and deferred work

Start with Discord direct messages, two allowlisted users, one service instance,
persistent storage, and a small approved capability set. Use a lightweight local
demonstration first, then exercise the same service interfaces across separate
machines.

Defer public registration, a custom web application, agent federation, unrestricted
tool installation, and broad autonomous exploration. Expand deployment complexity
only when actual usage requires it.

The immediate investment is the complete scientific interaction, dependable
execution records, human approval, and useful results. The connector implementation
lives in [chemgraph-connect](https://github.com/Autonomous-Scientific-Agents/chemgraph-connect).
