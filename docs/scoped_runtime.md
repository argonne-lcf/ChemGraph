# Scoped runtime for embedding services

`chemgraph.runtime` lets an application such as a lab service run the main
agent for authenticated callers. Every permitted tool call is reviewed against
an exact preparation, executed at most once, and its outputs are kept as
immutable artifacts. ChemGraph owns preparation, execution states and
artifacts. The application owns authentication, decision records, storage and
transport.

This is an application boundary, not an operating-system sandbox. Calculator
packages and model files are provisioned by the operator.

## Building and running a scoped agent

```python
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.agent.main_session import MainAgentSession
from chemgraph.registry.tools import ToolRegistry
from chemgraph.runtime import (
    ExecutionPolicy, FileOperationStore, OperationJournal, RunContext, run_context,
)

policy = ExecutionPolicy(
    calculators={"emt", "mace_mp"},
    model_roots=["/opt/models"],
    model_resources={"mace_mp:small": "/opt/models/mace-small.model"},
    limits={"max_steps": 500, "max_atoms": 200},
)
journal = OperationJournal(FileOperationStore("/srv/state/operations"))
await journal.recover()  # once at startup

context = RunContext(owner="alice", thread_id=thread, task_id=task,
                     workspace=f"/srv/state/workspaces/{thread}",
                     policy=policy, journal=journal)
agent = ChemGraph(
    workflow_type="main_agent", model_name="gpt-4o-mini",
    approval_mode="scoped",
    tool_registry=ToolRegistry(s for s in ToolRegistry().specs() if s.name in policy.tools),
    subagent_names=[], discover_skills=False, checkpointer=saver,
)
session = MainAgentSession(agent.workflow, thread_id=thread)
with run_context(context):
    turn = await session.run("Optimize water with EMT")
```

Keep one `RunContext` per conversation, change only `task_id` between
serialized turns, and give each conversation its own workspace. Scoped graphs
keep agent files in checkpoint state and expose no shell. They also have no
delegation and no approval interrupts other than the ones described below.

## What the model can call

- Framework helpers (`policy.helpers`: todos, checkpoint file reads, tool
  discovery, `ask_human`) run directly.
- Tools outside `policy.tools` return an error message to the model and do not
  run.
- Permitted tools go through review. The permitted set is the ASE tools that
  have an exact preparation: `run_ase`, the structure readers and writers, and
  the SMILES helpers.

Supported calculators are EMT and MACE (`mace_mp`, `mace_off`, `mace_anicc`).
A MACE model is used only through a `model_resources` binding to a weights file
under `model_roots`; model names alone are rejected. Calculator commands,
directories and model URLs are never accepted from tool arguments.

## Review

For each permitted call, `OperationJournal.propose` resolves defaults, copies
referenced input files to `.chemgraph-inputs/<sha256>/`, and saves the
preparation and its digest. If the call can't be prepared, for example because
its input doesn't exist yet, the model gets the reason without a review.

When a call needs a decision, the graph pauses with one interrupt per call:

```python
{"kind": "approval", "operation_id": ..., "tool_call_id": ..., "tool": ...,
 "digest": ..., "prepared": {...}}
```

Record the authenticated decision, then resume with any value:

```python
await journal.decide(operation_id, digest=digest, approve=True,
                     decided_by="alice", expires_at=time.time() + 600)
with run_context(context):
    turn = await session.resume("decided")
```

The resume value never authorizes anything; the middleware reads the decision
from the journal. An undecided or expired operation pauses again, including
one that is rejected or expires while it waits to run: the approval is checked
again atomically when `invoking` is saved. A rejection
reaches the model as an error message, and the decision's `message` explains
why. Execution uses the prepared arguments and snapshots, so later changes to
the original files cannot alter an approved action. If a snapshot or model
file changes after approval, the call is blocked and has to be proposed again.

## Execution states and recovery

```text
proposed -> approved -> invoking -> returned -> finalized
            rejected    uncertain    (artifacts published)
blocked                 abandoned
```

- `invoking` is saved before the tool runs and `returned`, with the result
  and the registered output paths, right after.
- An exception raised by the tool is a known outcome, so it is saved as an
  error result and finalized like any other result, with any outputs it
  registered.
- Finalization saves a checksum manifest of the registered outputs, then
  publishes exactly those files. If either step fails, the operation stays
  `returned`; retrying the turn finalizes it without running the calculation
  again.
- A finalized operation replays its saved result after verifying its
  artifact checksums.
- `journal.recover()` marks interrupted invocations `uncertain`. An uncertain
  or unfinished operation blocks further execution in that conversation until
  an operator reconciles it:
    - `confirm_nonexecution(id, decided_by=..., evidence=...)` returns it to
      review, so it needs a new approval;
    - `abandon(id, decided_by=..., reason=...)` stops it without claiming
      success.
- Every decision and reconciliation is kept in the record's `history`.

`OperationStore` is a small protocol (`get`, `put`, `records`) so a service can
keep operations in its own database. `put(record, expected_revision=...)` is a
compare-and-set on the record's `revision` and raises `OperationConflict` when
the stored revision differs (`None` means the record must not exist yet); a
conflicting `decide` raises `ValueError` so the operator can reload and decide
again. `FileOperationStore` stores one fsynced JSON file per operation and is
safe within one process only. Pass `authorizer=` to `OperationJournal` to add
authorization sources such as study grants.

## Artifacts

File-producing tools call `chemgraph.runtime.register_artifact(path)` after
writing. Outside a scoped invocation this does nothing.

Finalized tool messages carry artifact descriptors. All paths are relative to
the workspace, so a workspace restored to another location still replays:

```python
for artifact in result_artifacts(turn.state, task_id):
    path = artifact_file(context.workspace, artifact)
    # artifact: path, media_type, task_id, tool_call_id, source_path, sha256, size_bytes
```

Serve artifacts through the application's ownership checks rather than
exposing the workspace. Back up the workspace, the operation store and the
checkpoints together.
