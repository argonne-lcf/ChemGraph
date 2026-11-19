# Future direction: auto-resubmit via the IRI Facility API

This is a design note, not shipped code. It records how the cap-to-resume
mechanism validated in this example would extend into unattended, cross-allocation
auto-resubmission through the DOE IRI Facility API, and it explains why that
extension is purely additive on top of what already exists.

Nothing here is implemented. The goal is to give a reviewer the forward map so the
current PRs read as a deliberate first layer, complete in their own scope.

## Where auto-resubmit sits in the layering

The shipped feature is **Layer 1: enforcement**, and it runs entirely inside one
scheduler allocation (one `qsub` window on the compute node). A calculation that
would exceed the allocation's wall clock self-terminates at an ASE optimizer-step
boundary, writes a durable partial geometry plus a `run_manifest.json` with
`status="capped"` and a `pending_next_step`, and `chemgraph resume <session_id>`
continues from that partial. Within a single allocation this continuation is
automatic: the agent adopts the prior session's `log_dir`, clears the pending
step, and resumes from the partial with no human action. The examples here drive
that path either as one agent run that resumes itself or as a PBS script that runs
run1 then resume as two processes in the same allocation.

The step that stays manual today is starting the **next** allocation. When a whole
`qsub` window is exhausted, the compute node is released, so continuing the
calculation means submitting a fresh `qsub`. Nothing in ChemGraph submits a
scheduler job, so that next submission is done by a human running `qsub` again.

Auto-resubmit is **Layer 2: continuation across allocations**. A long-lived
service watches for capped manifests and submits the next allocation's resume job
itself, so a calculation that needs more wall time than any single allocation
grants runs to completion across many allocations with no human in the loop. This
maps directly to the IRI "Long-Term Campaign" science pattern, which names the
short-allocation / long-calculation mismatch as a target case.

Layer 2 consumes Layer 1's outputs. It adds no requirement back onto Layer 1.

## What the IRI Facility API provides

The IRI Facility API (ALCF reference implementation:
`github.com/argonne-lcf/alcf-facility-api`, forked from
`doe-iri/iri-facility-api-python`) is a REST service that standardizes job
submission and status across DOE facilities. A live instance runs at
`api.alcf.anl.gov`; NERSC and ESnet run their own. Each facility maps the standard
endpoints onto its local scheduler through a `FacilityAdapter` implementation.

The compute endpoints relevant to Layer 2:

| Purpose | HTTP | Path | operation_id |
|---------|------|------|--------------|
| Submit a job | POST | `/job/{resource_id}` | `launchJob` |
| Update a job | PUT | `/job/{resource_id}/{job_id}` | `updateJob` |
| Get one job's status | GET | `/status/{resource_id}/{job_id}` | `getJob` |
| List jobs | POST | `/status/{resource_id}` | `getJobs` |
| Cancel a job | DELETE | `/cancel/{resource_id}/{job_id}` | `cancelJob` |

A submission carries a scheduler-agnostic `JobSpec`. The fields that matter for a
resume job, and their source in this example's PBS scripts:

| JobSpec field | Meaning | Source in our PBS script |
|---------------|---------|--------------------------|
| `executable` + `arguments` | what to run | `chemgraph resume <session_id>` |
| `environment` (dict) | env vars for the job | the `export` block (`ESPRESSO_PSEUDO`, proxy vars, the LLM model) |
| `directory` | working directory | the per-job work dir |
| `stdout_path` / `stderr_path` | output capture | `#PBS -j oe` target |
| `launcher` | `mpirun` / `srun` | the `mpiexec` in `ASE_ESPRESSO_COMMAND` |
| `resources.node_count` / `process_count` | node/rank counts | `#PBS -l select`, `NRANKS` |
| `attributes.duration` | wall time in seconds | `#PBS -l walltime` |
| `attributes.queue_name` | queue / partition | `#PBS -q debug` |
| `attributes.account` | project to charge | `#PBS -A ChemGraph` |

Status comes back as a `JobStatus` with a PSI-J-derived `state` enum
(`new` / `queued` / `held` / `active` / `completed` / `failed` / `canceled`), a
single `time` timestamp (seconds since epoch), and an `exit_code`.

## How Layer 2 would use it

A login-node service (the compute node is gone once a job caps, so submission has
to originate somewhere persistent) would:

1. Poll for `run_manifest.json` files whose `status` is `capped`.
2. Read the `session_id` and the `pending_next_step` (whose `input_structure_file`
   is the saved partial geometry).
3. Build a `JobSpec` with `executable="chemgraph"`,
   `arguments=["resume", session_id]`, the environment and resource fields copied
   from the original submission, and `attributes.duration` set to a fresh
   allocation window.
4. POST it to `launchJob`.
5. Poll `getJob` until the state is terminal, then repeat from step 1 if the new
   run capped again.

The `chemgraph resume <session_id>` path already exists and is validated in this
example. Layer 2 only chooses when and where to invoke it.

### How the two layers fit together

It may help to state the division of labour, since the within-allocation half is
already in place. The agent layer (validated here on real QE) handles everything
inside a single allocation: it caps at a step boundary, writes the partial and
manifest, and on resume adopts the prior `log_dir`, clears the pending step, and
continues. The launcher layer described above would handle only the transition to
the next allocation: notice a capped manifest and submit the follow-on job. The
two communicate through three seams that already exist, so the launcher would not
need to touch the agent or calculator code:

- **detection**: the manifest's `status="capped"` field tells the launcher a
  follow-on is due;
- **continuation key**: `session_id` is all the launcher needs to pass on, since
  resume resolves the `log_dir`, partial geometry, and pending step from it;
- **time authority**: each allocation sets its own
  `CHEMGRAPH_ALLOCATION_DEADLINE`, so the launcher supplies a fresh window per
  submission and the agent enforces it.

The IRI Facility API is one way to implement the launcher, and a convenient one
where portability across facilities matters. A simpler login-node loop that calls
`qsub` directly would use the same three seams. Either way, the choice is confined
to the launcher layer and leaves the shipped enforcement path unchanged.

## Why this is additive, with no breaking change to the current PRs

Three properties keep Layer 2 outside the scope of the shipped work:

1. **The manifest is forward-compatible by construction.** `run_manifest.json`
   carries a `schema_version` (currently 1) and is read with a permissive JSON
   loader that tolerates unknown keys. Any field Layer 2 might want later (a
   facility job id, a submission timestamp, an `iri_*` block) can be added without
   breaking an older reader. None of those fields are needed now.

2. **Our deadline stays authoritative.** `JobStatus` reports a state and one
   timestamp; it exposes no live "seconds remaining". The cap already computes its
   own deadline from `CHEMGRAPH_ALLOCATION_DEADLINE` (absolute epoch) minus
   `CHEMGRAPH_ALLOCATION_MARGIN`, read fresh from the environment on each run. That
   self-computed deadline remains the single source of truth whether or not a
   Facility API sits above it, so adopting IRI changes nothing in the enforcement
   path.

3. **The join key is already `session_id`.** Resume finds the prior run's
   `log_dir` (and thus its manifest and partial geometry) through the session
   store, keyed by `session_id`. The facility's own `Job.id` is bookkeeping for
   the resubmitter, so it belongs to Layer 2 and never has to enter the manifest.

## What Layer 2 additionally requires (and why it is deferred)

These are the reasons Layer 2 is a separate future effort held apart from the
current validation:

- **A non-demo facility adapter.** The reference implementation ships a demo
  adapter that returns fake data; a real submission needs the facility's adapter
  wired to the live scheduler behind `api.alcf.anl.gov`.
- **A separate Globus scope.** The Facility API authorizes through Globus and
  needs an `iri-api` scope. That is distinct from the inference scope this
  example's LLM calls already use, and it requires the account to be registered
  with the IRI resource server.
- **A persistent login-node service.** Submission has to run somewhere that
  outlives the capped compute-node process.

Because none of these can be exercised without live facility access, Layer 2 would
be developed the same way VASP support is: written against the documented
interface and unit-tested hermetically, with a clear label that it has not been
run against the live API. The validated real-DFT results in this example stay
scoped to Layer 1 (the cap-to-resume mechanism on real Quantum ESPRESSO).

## Relationship to other layers already in the codebase

Layer 2 is also distinct from the Parsl / Academy launcher path. That launcher is
the natural place to export `CHEMGRAPH_ALLOCATION_DEADLINE` into each worker so the
cap knows its window; the Facility API is the transport that resubmits a fresh
allocation after a cap. They compose (a launcher sets the deadline; the Facility
API submits the next allocation) and neither one changes the enforcement code.
