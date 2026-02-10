!!! note
    ChemGraph exposes tools through Model Context Protocol (MCP) servers in `src/chemgraph/mcp/`.

## Available servers

- `mcp_tools.py`: general ASE-powered chemistry tools
- `mace_mcp_parsl.py`: MACE + Parsl workflows
- `graspa_mcp_parsl.py`: gRASPA + Parsl workflows
- `data_analysis_mcp.py`: analysis utilities for generated results

## Run a server

### stdio transport (default)

```bash
python -m chemgraph.mcp.mcp_tools
```

### streamable HTTP transport

```bash
python -m chemgraph.mcp.mcp_tools --transport streamable_http --host 0.0.0.0 --port 9003
```

## Common CLI options

All MCP servers use:

- `--transport` with `stdio` or `streamable_http`
- `--host` for HTTP mode
- `--port` for HTTP mode

## Docker mode

You can run MCP server mode with Docker Compose:

```bash
docker compose --profile mcp up
```

Endpoint: `http://localhost:9003`

## Notes for Parsl-based servers

`mace_mcp_parsl.py` and `graspa_mcp_parsl.py` rely on Parsl and HPC-specific configuration. Ensure your environment is prepared for the target system before running production jobs.
