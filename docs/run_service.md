# Python run service

`chemgraph.service.RunService` provides an asynchronous Python API for persistent
conversations and local ASE calculations. It uses `main_agent` with direct chemistry
tools, exact operation reviews, SQLite checkpoints, and a persisted local job queue.
This first milestone supports the offline EMT calculator. FastAPI is the next
milestone: an HTTP adapter over these methods. Authentication and clients are
separate follow-ups.

```python
from chemgraph.service import RunService

async with RunService("./chemgraph-data", model_name="gpt-4o-mini") as api:
    session = await api.create_session()
    run = await api.start_run(session.id, "Build H2 and calculate its EMT energy.")
    while True:
        run = await api.wait(run.id)
        if run.status != "waiting_for_user":
            break
        for pending in run.interrupts:
            if pending.kind == "approval":
                print(pending.payload["prepared"])
                accepted = input("Approve this exact operation? [y/N] ") == "y"
                await api.decide(run.id, pending.operation_id,
                                 digest=pending.payload["digest"], approve=accepted)
            elif pending.kind == "clarification":
                answer = input(str(pending.payload) + " ")
                await api.respond(run.id, pending.id, answer)
    print(run.status, run.assistant_response, run.error)
    for artifact in await api.list_artifacts(run.id):
        print(await api.resolve_artifact(run.id, artifact.id))
```

Provider credentials follow the existing model loader configuration. For a custom
model, supply `model_factory(session_id)` and a stable, non-secret
`configuration_id`. Supply the same configuration when reopening the data root.
The runnable `examples/run_service_local.py` uses a scripted model and EMT with
no credentials, model downloads, or network access.

## API and states

The context manager owns its data root exclusively; a second service cannot open
the same root concurrently. `list_sessions()`, `get_session(id)`, and `get_run(id)`
return typed snapshots. `start_run(session_id, message)` allocates a stable run ID
for one conversation turn. Approval, job waits, and retries retain that ID.
Only one run may advance per session. Start a follow-up after completion.

`wait(run_id)` returns when the run completes, fails, requires recovery, is
cancelled, or needs human input. Job waits resume automatically after publication.
Use `respond` only for current clarification interrupts and `decide` for current
operation approvals. A response cannot approve a tool or supply a job result.
Parallel interrupts are combined into one graph resume after all are ready.
`on_event` accepts the existing main-agent event callback; persisted snapshots
remain authoritative.

`retry_run(run_id)` resumes a failed graph checkpoint without adding another user
message. A failed calculation is a tool error that the agent can inspect; it does
not necessarily fail the conversation.

`cancel_run(run_id)` makes the entire session terminal. It cancels queued jobs and
attempts to cancel active work. A running process may finish later; its outputs
are archived with `promoted=False`, without updating logical session paths or
resuming the graph. Create a new session for subsequent work. `list_jobs(run_id)`
reports actual calculation states separately from the run's cancellation state.

## Persistence and recovery

The data root contains service metadata, readable history, checkpoints, operation
records, job receipts, and session workspaces. Reopening restores pending reviews
and queued jobs. Dispatch intent is durable before process submission. A previously
dispatched job without a valid completion envelope becomes `unknown`; it is never
automatically executed again. Its run becomes `recovery_required`.

After investigating an uncertain operation, call
`reconcile(run_id, operation_id, action="confirm_nonexecution", evidence="...")`
only when you can establish that it did not run. This creates a fresh attempt and
requires a new review, retaining the original input snapshot. Confirmation is
rejected while the worker is known live or has a completion envelope. Alternatively,
`action="abandon"` records the decision to stop waiting; it does not claim that a
running process stopped. Recovered failed graph checkpoints can then be retried.
`reconcile` returns the updated run snapshot, like the other run mutations.

Graceful close stops dispatching new work, waits for active calculations, archives
their results, and preserves outstanding waits for reopening. It is not a bounded
shutdown or process termination API.

## Files and artifacts

Each job uses a private attempt directory and reviewed input snapshots. After a
successful publication, ordinary session paths such as `energy.json` and
`h2_opt.traj` are available to subsequent chemistry tools. Jobs execute in a global
FIFO queue with one local worker; later publications overwrite matching logical
names. Each operation also has independent immutable artifacts, so previous
results remain available. Calculator cleanup removes obsolete logical side files
without deleting prior artifacts.

`resolve_artifact(run_id, artifact_id)` verifies the saved size and digest before
returning a local path. Missing or modified artifacts raise an error. The reserved
`.chemgraph-jobs`, `.chemgraph-inputs`, and artifact trees are internal execution
state; ordinary tools cannot use them as arbitrary file paths.
