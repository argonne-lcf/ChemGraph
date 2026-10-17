# Persistent MCP service

The opt-in service hosts the scoped chemistry catalog for a trusted research
group. Each user has a stable identity, private workspace, durable operation
history, and immutable artifacts. ASE calculations run in separate supervised
processes. A calculation crash does not break another calculation's worker.
Models load for each calculation; no model cache is shared between processes.

This is application-level authorization under one OS account, not a sandbox for
untrusted code. Only operator-permitted calculators and installed models run.
Existing standalone MCP servers and stdio clients retain their current behavior.
The service auto-approves calls allowed by its policy; it does not ask a human to
review each operation. EMT is useful for integration tests, not a default research
method for arbitrary chemistry.

## Start on one managed host

Use a POSIX host and the normal ChemGraph environment. No new package extra is
required. Start with [the example configuration](../deploy/mcp/service.toml),
changing `data_dir` and `public_url`. The URL is an origin, without `/mcp`.
Use HTTPS except for loopback development. Put the data directory on local disk:
SQLite WAL is not supported on NFS. Keep the directory private to the service
account and do not mount it into client environments.

```bash
chemgraph mcp --config /etc/chemgraph/service.toml token issue alice
chemgraph mcp --config /etc/chemgraph/service.toml serve
```

The first command prints a new bearer token once. Deliver it privately to Alice;
store it in her environment or secret manager, not in a repository. Token hashes,
expiry and revocation are stored in `service.sqlite3`. Tokens expire after 30 days
by default; `issue` and `rotate` accept `--days`.

```bash
chemgraph mcp --config /etc/chemgraph/service.toml token list
chemgraph mcp --config /etc/chemgraph/service.toml token rotate TOKEN_ID
chemgraph mcp --config /etc/chemgraph/service.toml token revoke TOKEN_ID
```

Rotation invalidates the old token while preserving the stable user ID and all
existing data. A name identifies the user for operator administration; filesystem
paths use generated user IDs. The token-management CLI is local administration
and is not exposed as an HTTP route or MCP tool.

Use [the systemd unit](../deploy/mcp/chemgraph-mcp.service) and
[Caddy example](../deploy/mcp/Caddyfile) for a supervised deployment. Provision the
`chemgraph` account and the Python environment at the paths in the unit first.
Caddy terminates TLS and preserves the public Host header; the application checks
that host and allowed origins explicitly. The example also caps request bodies.
See Caddy's [reverse proxy documentation](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy).

Run exactly one service process. The application takes an exclusive instance
lock. If adapting the existing Kubernetes manifest, use **one replica**, replace
its standalone-server command, and supply local persistent storage and a proxy.
Do not reuse the existing unauthenticated LoadBalancer deployment unchanged.
The application ignores forwarded identity and scheme headers; the configured
public origin and bearer token establish its boundary.

`/healthz` is a minimal public liveness probe. `/readyz` returns 200 only after
reconciliation and dispatcher startup, otherwise 503. Probes must send the
configured Host header. Every other route requires a bearer token, including
artifact routes, slash redirects and tool discovery. There is no OAuth issuer or
authorization metadata in this pilot: configure clients with the bearer header.

## Connect a client

Set `CHEMGRAPH_SERVICE_TOKEN` privately, then use:

```bash
chemgraph run --mcp-url https://chemgraph.example/mcp \
  --mcp-token-env CHEMGRAPH_SERVICE_TOKEN -w single_agent \
  -q "Use the MCP tools to prepare a water structure."
```

Or use the normal client configuration:

```toml
[mcp]
url = "https://chemgraph.example/mcp"
token_env = "CHEMGRAPH_SERVICE_TOKEN"
```

Only the variable's name goes in TOML. An explicitly configured missing variable
fails before connecting. Other MCP clients can supply
`Authorization: Bearer <token>` on their HTTP connection. Automatic OAuth login
is not implemented.

## Submit chemistry and retrieve artifacts

The service exposes the scoped tools documented in [scoped runtime](scoped_runtime.md),
plus optional `molecule_name_to_smiles`, job tools and artifact tools. All chemistry
calls pass through preparation and policy checks. Generic shell commands, arbitrary
model weights, specialized HPC servers and Globus Transfer are not exposed.

An MCP agent can call `put_workspace_file(path="water.xyz", content="...")` to
write small UTF-8 `.xyz`, `.cif` or `.json` inputs. It can also prepare coordinates
with the chemistry tools. Paths are workspace-relative. Inputs can be referenced
by a returned `artifact:<id>` instead of a mutable workspace filename.

`run_ase` accepts the existing ASE parameters, requires an explicitly permitted
calculator (unless the operator sets a policy default), and returns `submitted`
with a `batch_id`. It snapshots the input into a private job directory before
queuing. Input bytes, prepared arguments, model identity and policy are checked
again before invocation. Structure readers accept XYZ, extended XYZ, CIF, JSON
and ASE trajectory files; client-supplied calculator commands are forbidden.

Use `check_job_status`, `list_jobs`, `get_job_results` and `cancel_job`. Status is
the canonical runtime state:

| State | Meaning |
| --- | --- |
| `approved` | Queued and eligible for dispatch. |
| `invoking` | Invocation intent saved; a worker may have started. |
| `returned` | Outcome saved; artifact publication remains. |
| `finalized` | Outcome and artifacts saved. Check `outcome` and scientific convergence. |
| `uncertain` | Execution could not be confirmed; never automatically repeated. |
| `blocked` | Preparation or revalidation failed before invocation. |
| `abandoned` | Withdrawn before dispatch, or explicitly abandoned by an operator. |

An observed tool exception is a finalized error outcome, not an uncertain result.
Terminating a running calculation produces `uncertain`, since partial work may
have occurred. Canceling a queued operation abandons it without execution. Tool
results include artifact references and authenticated download paths. Use
`read_artifact` for small UTF-8 results and `list_artifacts` for bounded discovery.
Lists accept `limit` (1–100) and `offset`.

For larger/binary data, scripts can send a raw HTTP request body to
`POST /artifacts?path=filename` and fetch `GET /artifacts/<id>`, using the same
bearer token. These are not multipart endpoints. Unknown and foreign artifact/job
IDs produce identical responses. Published artifacts are immutable; uploading to
an existing workspace filename does not overwrite earlier artifact versions.

## Limits and operations

Defaults are four calculation processes globally, one per user, ten queued
calculations per user, and four short-running tool/transfer operations. Eligible
users receive turns in round-robin order. Each calculation has a one-hour wall
limit. Increase limits deliberately for the host and chosen calculator; configure
GPU concurrency to fit available device memory.

Text artifacts are limited to 1 MiB, HTTP uploads to 100 MiB, and per-user stored
files to 10 GiB. MCP request bodies are bounded at eight times the text limit to
allow JSON encoding overhead. Streaming uploads are counted as they arrive.
Storage accounting includes job inputs, snapshots, logs and immutable artifacts;
running jobs are monitored and individual worker files are bounded. These are
application limits, not hard OS memory/filesystem quotas. Provision host limits
appropriate for the group. Only the parent service writes operation records.

When enabled, PubChem requests share the service host's outbound path and upstream
rate limits. Set `enable_pubchem = false` to remove that tool, as in the sample
configuration. No network or model downloads are required by the test suite.

Logs identify requests, users, jobs and child PIDs without logging bearer headers.
Per-job native stdout/stderr remain private operator files. Workers receive a
minimal environment and no token store connection or client credentials. Short
calls use threads with the same workspace confinement; they are not a native-code
crash sandbox.

## Recovery, backup and rollout

On startup, `approved` work is revalidated and queued, `invoking` becomes
`uncertain`, and `returned` work finishes artifact publication without recalculating.
Catalog registration is idempotent after `finalized`; an interrupted registration
is completed at startup or by the next `get_job_results`. If publication fails at
startup, the failure is logged, only that job stays `returned`, and the service
starts for other users. Repair storage or increase the quota, then call
`get_job_results` again.

The invocation-intent commit precedes process creation. A crash in between can
therefore leave an uncertain job that never ran. Do not infer nonexecution from
missing output alone. The operator can reconcile using evidence:

```bash
chemgraph mcp --config /etc/chemgraph/service.toml job abandon JOB_ID --reason "Outcome cannot be established"
chemgraph mcp --config /etc/chemgraph/service.toml job confirm-nonexecution JOB_ID --evidence "Recorded evidence that the child never started"
chemgraph mcp --config /etc/chemgraph/service.toml job approve JOB_ID
```

Confirmation returns an uncertain operation to review; the separate approve
command validates its existing preparation and explicitly permits another attempt.
Changed inputs, model files or policy require a new submission.

Use a supervisor that kills old calculation processes before restarting the
parent; the supplied unit uses `KillMode=control-group`. Stop accepting work,
allow the configured shutdown grace, then terminate remaining calculation groups.
The systemd stop timeout also bounds shutdown of a stuck native thread. Do not
manually kill only the parent and launch a replacement while children remain.

For a consistent backup, stop the service and copy the **entire data directory**,
including SQLite files and any WAL sidecars, workspaces, job directories and
artifact files. Restore the complete directory with private permissions, restore
the same operator models/configuration, and start one instance. Do not copy a live
SQLite main file alone. Tokens are part of the backup; rotate them if a backup is
exposed. A restored finalized result must be retrieved without executing chemistry.

Start with a new service database. Existing ownerless JSON job trackers are not
imported automatically. Smoke-test two users with EMT, verify foreign-ID rejection,
then enable the research group's validated calculator/model explicitly. Existing
standalone-server users can continue using their previous commands.
