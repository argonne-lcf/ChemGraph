# Test a persistent MCP deployment

This example runs two real ASE/EMT energy calculations through the official MCP
Python client. It checks authentication, user isolation, reconnects, uploaded input
bytes, finite energies in eV, and downloaded artifact sizes and SHA-256 hashes.
It needs the normal ChemGraph installation (`pip install -e .`), Python 3.11+, and
no LLM credentials or model downloads. EMT is a smoke-test calculator, not a
validated research method for water.

## Temporary local service

From the repository root, using the Python environment containing ChemGraph:

```bash
python examples/persistent_mcp/smoke_test.py local
```

On a POSIX host this creates a private temporary configuration, issues tokens for
two users, and starts Uvicorn on loopback. It submits both jobs, disconnects and
reconnects, then cleanly stops and restarts the service with the same data. The
second check only retrieves saved results and requires unchanged timestamps and
checksums. Successful output contains two `PASS:` lines. Exit status is zero on
success and nonzero on failure. `--timeout 180` changes the per-job deadline;
local startup is capped at 30 seconds and individual requests at 30 seconds.

Normal completion removes the temporary directory only after this server's log
confirms application shutdown. A crash, forced shutdown or unconfirmed cleanup
fails the run, skips restart, and retains the directory and `server.log`; the
error prints its location. Calculation processes have independent process groups
and may survive killing the parent. Stop any remaining workers before cleaning
up or reusing that directory. Graceful shutdown is allowed 15 seconds with two
workers and a configured two-second service grace period.

## Existing HTTPS service

Configure the [persistent service](../../docs/persistent_mcp.md), including its
[systemd unit](../../deploy/mcp/chemgraph-mcp.service) and
[Caddy proxy](../../deploy/mcp/Caddyfile). Use the public HTTPS origin below,
without `/mcp`. The policy must allow `run_ase` with the `emt` calculator, three
atoms and at least one step. The script explicitly requests `steps=1`.

Use dedicated test users: remote runs leave persistent uploads, jobs and
artifacts, and the service has no deletion API. The operator can issue their
tokens using the service configuration on the server:

```bash
export CHEMGRAPH_ALICE_TOKEN="$(chemgraph mcp --config /etc/chemgraph/service.toml token issue smoke-alice)"
export CHEMGRAPH_BOB_TOKEN="$(chemgraph mcp --config /etc/chemgraph/service.toml token issue smoke-bob)"
```

For a separate client host, deliver those values privately through your normal
secret-management process. Do not put token values in command arguments or files
in the repository. The script takes only environment variable names:

```bash
python examples/persistent_mcp/smoke_test.py remote \
  --url https://chemgraph.example \
  --alice-token-env CHEMGRAPH_ALICE_TOKEN \
  --bob-token-env CHEMGRAPH_BOB_TOKEN \
  --receipt smoke-receipt.json
```

The two tokens must identify different users. Before any calculation, the script
checks that Bob cannot retrieve Alice's upload. Failure means the tokens may
identify the same user or isolation is broken; there is no identity-discovery
endpoint. Both users then upload different water structures under the same
relative filename, using MCP text upload and raw HTTP upload respectively.

The receipt is exclusively created **before any network request**; an existing
file is never overwritten. It contains no credentials. Partial progress records
the run prefix and each returned upload/job ID for operator accounting, even if
later checks fail. A failed run can still leave data on the server; a lost
response can leave an unrecorded operation, so retain the printed run prefix and
consult server logs as needed. Verify rejects incomplete receipts and never
submits replacement work. Use a new receipt filename for each new remote run;
retain receipts for as long as their referenced data is needed. Do not modify
the live database to clean up test data.

After an operator-controlled restart using the configured supervisor:

```bash
sudo systemctl restart chemgraph-mcp
python examples/persistent_mcp/smoke_test.py verify \
  --url https://chemgraph.example \
  --alice-token-env CHEMGRAPH_ALICE_TOKEN \
  --bob-token-env CHEMGRAPH_BOB_TOKEN \
  --receipt smoke-receipt.json
```

Run the restart command on the service host and the verification command on the
client. Verification checks saved results, timestamps and artifact checksums;
it does not prove that an operator actually restarted the remote process.

TLS certificate verification stays enabled and redirects are disabled. HTTP is
accepted only for loopback development. Receipt origins must match `--url`, and
artifact paths must remain on that explicit origin. The script checks readiness,
missing credentials and disallowed Origin responses. The wrong-Host probe runs
only against the local Uvicorn process: remote/verify modes explicitly report it
as untested because a reverse proxy may intercept it. The example's automated
validation covers local Uvicorn and the ASGI app; an actual HTTPS/Caddy deployment
must be checked in your environment.
