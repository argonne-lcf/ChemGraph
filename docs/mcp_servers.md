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

## MLIP server

The MLIP server exposes `run_mlip` and `run_mlip_batch` through the configured
execution backend:

```bash
python -m chemgraph.mcp.run_mlip_mcp
```

Its request schema has one calculation envelope. The model identifies the
scientific potential, while `calculator.backend` chooses the adapter that
evaluates it. For example, ASE and NVIDIA ALCHEMI use the same MACE model
shape:

```json
{
  "params": {
    "input_structure_file": "/data/water.xyz",
    "output_results_file": "/data/water-result.json",
    "driver": "energy",
    "model": {
      "family": "mace",
      "checkpoint": "medium"
    },
    "calculator": {
      "backend": "ase",
      "device": "cpu"
    }
  }
}
```

To route that scientific request through ALCHEMI, only the calculator object
changes:

```json
{
  "calculator": {
    "backend": "nvalchemi",
    "device": "cuda",
    "dtype": "float32",
    "compile_model": false,
    "enable_cueq": false
  }
}
```

Rootstock is also selected as a calculator rather than represented as a model
family:

```json
{
  "model": {"family": "uma", "checkpoint": "organization/model"},
  "calculator": {
    "backend": "rootstock",
    "cluster": "local",
    "device": "cuda"
  }
}
```

The server-wide `[execution]` configuration independently controls where the
whole tool call runs. It is not repeated inside the MLIP request:

```toml
[execution]
backend = "parsl" # local, parsl, ensemble_launcher, or globus_compute
system = "polaris"
```

`run_mlip_batch` submits the whole ordered batch as one execution task. The
calculator/model is therefore loaded once and reused; `batch_size` and
`max_atoms` control internal ALCHEMI chunks, not execution-backend fan-out.
CUDA ASE and ALCHEMI requests add a one-GPU advisory resource hint. Backends
may ignore advisory hints.

With a non-shared filesystem such as a remote Globus Compute endpoint, the
server does not copy or inline MLIP input files. Pre-stage structures, local
checkpoints, and Rootstock weights (the transfer tools are exposed when Globus
Transfer is configured), pass their remote paths, and use an absolute remote
output path. Remote submissions return a `batch_id`; use `check_job_status` and
`get_job_results` to retrieve the result.

## Artifacts and security

Set `CHEMGRAPH_LOG_DIR` before starting a server to choose its log/artifact
directory. Keep server and client filesystem visibility in mind across hosts or
containers. Restrict network access, use least-privilege credentials, validate
inputs, and avoid mounting sensitive directories into the server process.
