# Live HPC batch demo

Run from a POSIX submitting host with Python 3.11+ and ChemGraph installed
(`python -m pip install -e .`). The script loads this checkout's source even
if another worktree is installed in your environment.
Keep the full checkout available while running or collecting; a temporary
directory may be cleaned between sessions. If the script reports an incomplete
checkout, restore its missing files and retry `collect` with the same config and
run directory. Preserve the config and all run evidence during recovery.

This example exercises the native `hpc_*` tools with real Globus Transfer and
ALCF IRI services. It submits one PBS job, adds the integers 1 through 10 on
the allocated node, and retrieves the result, compute hostname and full job ID.
It verifies the input SHA-256 and the expected sum of 55. No LLM, remote Python
environment or model download is needed; the job uses Bash, awk and sha256sum.
This demonstrates the batch lifecycle rather than a chemistry calculation.
The launch script enters its own staged directory before reading inputs, since
PBS may start it in the user's home directory.

## Configure and authenticate

From the repository root:

```bash
cp examples/hpc_batch/config.example.toml examples/hpc_batch/config.local.toml
```

Edit `config.local.toml` and replace every `YOUR_*` value. The local copy is
ignored by Git. Create the configured local and remote root directories first.
The two views of each directory must refer to the same storage:

| Field | Meaning |
| --- | --- |
| `local_root` | Absolute path on the machine running this demo |
| `local_collection_root` | That directory's path in the local Globus collection |
| `remote_root` | Absolute path visible to the compute nodes |
| `remote_collection_root` | That directory's path in the remote Globus collection |

For example, Eagle's compute path `/lus/eagle/projects/PROJECT/USER/demo`
may appear as `/PROJECT/USER/demo` through its collection. Check both paths
for your collection; they need not be identical. For Globus Connect Personal,
allow access to the local root, including `.hpc-inputs` and retrieved files.
Keep Globus Connect Personal running and connected during both transfers;
`ExternalError.DirListingFailed.GCDisconnected` means its host is offline.

The sample requests one node for 300 seconds in Polaris's `debug` queue,
consistent with the [Polaris queue limits](https://docs.alcf.anl.gov/polaris/running-jobs/).
Set a project allocation you can charge. This demo limits requests to one
node and at most 300 seconds; queue waiting is separate from job walltime.
`setup_script` and `python_executable` are unnecessary for this shell job.

Authenticate in your terminal using the same Python environment:

```bash
python -m chemgraph.execution.globus_transfer --collection YOUR_REMOTE_COLLECTION_UUID
python -m chemgraph.tools.alcf_iri_core login
```

The `--collection` option requests access to the remote mapped collection;
do not use it for a Globus Connect Personal collection. Sign in with an identity
allowed by the facility. Tokens stay in the normal local caches, outside TOML.
This example loads `[hpc]` itself; the general ChemGraph CLI does not yet load
these target settings automatically.

## Run

Check collection listing access and authenticated scheduler access first:

```bash
python examples/hpc_batch/run_demo.py check \
  --config examples/hpc_batch/config.local.toml --target polaris
```

This does not submit a transfer or a job. It cannot prove write permission,
allocation balance, or that the compute path maps to the intended collection.

Choose a **new** directory directly beneath your configured `local_root`:

```bash
python examples/hpc_batch/run_demo.py run \
  --config examples/hpc_batch/config.local.toml --target polaris \
  --run-dir /absolute/local_root/demo-001
```

`run` authorizes real transfers and one job charged to the configured project.
It prints the effective request before sending it. Direct Python tool invocation
does not run an agent approval graph; `allow_unreviewed_mutations = false`
still preserves mandatory reviews when these tools are registered in an agent.

The script prints JSON progress records with transfer IDs, remote directory,
submission evidence and scheduler status. Success ends with `"step": "verified"`
and a zero exit code. Results are in `demo-001/retrieved/`. All local and remote
files are retained, including `run.json`, `submission.json` and
`submission.started`. Do not edit or delete that evidence.

## Inspect an interrupted or slow run

The default wait deadline is 600 seconds per transfer or job (including queue
time). Increase it with `--timeout 1800`; `--poll-interval` defaults to 5 seconds.
A timeout does not cancel a transfer or job. The script refuses to `run` in an
existing directory and never automatically resubmits a job.

```bash
python examples/hpc_batch/run_demo.py status \
  --config examples/hpc_batch/config.local.toml --run-dir /absolute/local_root/demo-001
python examples/hpc_batch/run_demo.py collect \
  --config examples/hpc_batch/config.local.toml --run-dir /absolute/local_root/demo-001
```

`collect` waits for the recorded job and retrieves/verifies its outputs without
submitting computation. If a retrieval task was already recorded, it polls that
same task. Unknown transfer outcomes require manual inspection of the saved
submission ID and the native tools' replay instructions. A rejected or unknown
scheduler submission must be inspected in this directory; starting a new demo
is not a recovery procedure. On a failed job, inspect its saved status and use
the native retrieval tool to obtain stdout/stderr for diagnosis.
