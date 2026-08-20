# MCP servers

ChemGraph can expose chemistry tools to Model Context Protocol clients or load
tools from an MCP server into a ChemGraph agent.

## General chemistry server

The general server provides molecule-name lookup, SMILES-to-3D conversion, ASE
calculations, and result extraction.

### Stdio

```bash
python -m chemgraph.mcp.mcp_tools
```

A client configuration launches that command directly, for example:

```json
{
  "mcpServers": {
    "chemgraph": {
      "command": "python",
      "args": ["-m", "chemgraph.mcp.mcp_tools"]
    }
  }
}
```

The client and server must use the same Python environment. Server logs go to
stderr so stdout remains the JSON-RPC channel.

### Streamable HTTP

```bash
python -m chemgraph.mcp.mcp_tools \
  --transport streamable_http \
  --host 127.0.0.1 \
  --port 9003
```

The endpoint is `http://localhost:9003/mcp/`. The `/mcp/` path and trailing
slash matter. Binding to `127.0.0.1` keeps the server local; do not expose an
unauthenticated tool server to an untrusted network.

## Load MCP tools into ChemGraph

```bash
chemgraph run \
  --mcp-url http://localhost:9003/mcp/ \
  -q "Build a coordinate file for methane."
```

Or put one connection in `config.toml`:

```toml
[mcp]
url = "http://localhost:9003/mcp/"
server_name = "ChemGraph General Tools"
```

For stdio, use `--mcp-command` or the matching config key. Run
`chemgraph run --help` for installed-version syntax.

## Specialized servers

The source tree also contains advanced servers for ASE and MACE HPC jobs,
gRASPA and XANES direct/Parsl jobs, data analysis, HPC utility operations, and
file transfer. These are deployment building blocks, not zero-configuration
public services. They can require ALCF filesystems, scheduler configuration,
Parsl, external executables, and credentials.

Start from the [general examples](https://github.com/argonne-lcf/ChemGraph/tree/main/scripts/mcp_example),
[Parsl example](https://github.com/argonne-lcf/ChemGraph/tree/main/scripts/mcp_parsl_example),
or [XANES examples](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/xanes_mcp).

## ALCF IRI Facility API server

`chemgraph.mcp.alcf_iri_mcp` exposes ALCF's IRI Facility API
(<https://api.alcf.anl.gov>) as 43 flat MCP tools -- one per endpoint,
named `alcf_<category>_<action>`. Public endpoints (machine status,
incidents, facility metadata) work with no auth; project/allocation, PBS
job, and filesystem endpoints need a Globus-issued ALCF token. Write
actions (`submit_job`, `cancel_job`, `mkdir`, `rm`, `chmod`, ...)
additionally require `ALCF_IRI_ALLOW_UNSAFE=1` in the server's
environment.

```bash
python -m chemgraph.mcp.alcf_iri_mcp                                 # stdio
python -m chemgraph.mcp.alcf_iri_mcp --transport streamable_http --port 9010
```

Same underlying implementation as the `single_agent_iri` LangGraph
workflow (see [`examples/iri/`](https://github.com/argonne-lcf/ChemGraph/tree/main/examples/iri)),
so behaviour and auth flow are identical -- this server just lets
non-LangChain MCP clients reach the same tools. A capability card for
`main_agent`-style skill routing lives at
[`src/chemgraph/skills/alcf_iri.md`](https://github.com/argonne-lcf/ChemGraph/blob/main/src/chemgraph/skills/alcf_iri.md).

## Artifacts and security

Set `CHEMGRAPH_LOG_DIR` before starting a server to choose its log/artifact
directory. Keep server and client filesystem visibility in mind across hosts or
containers. Restrict network access, use least-privilege credentials, validate
inputs, and avoid mounting sensitive directories into the server process.
