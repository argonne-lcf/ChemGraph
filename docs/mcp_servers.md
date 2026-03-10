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

## Using with OpenCode

ChemGraph MCP tools can be used directly with [OpenCode](https://opencode.ai), giving you an AI coding agent with access to molecular simulation capabilities.

### Quick start

1. Copy the example configuration:

    ```bash
    cp .opencode/opencode.example.jsonc opencode.json
    ```

2. Set `CHEMGRAPH_PYTHON` to your ChemGraph Python interpreter:

    ```bash
    # Option A: a project-local venv
    export CHEMGRAPH_PYTHON=env/chemgraph_env/bin/python

    # Option B: a standard venv
    export CHEMGRAPH_PYTHON=.venv/bin/python

    # Option C: whatever environment is currently active
    export CHEMGRAPH_PYTHON=$(which python)
    ```

    !!! tip
        Add the export to your shell profile (`~/.bashrc`, `~/.zshrc`) so you don't have to set it every time.

3. Launch OpenCode:

    ```bash
    opencode
    ```

    The `chemgraph` MCP tools (molecule lookup, structure generation, ASE simulations) will be available automatically.

### Available MCP servers for OpenCode

The example config (`.opencode/opencode.example.jsonc`) includes all servers. Enable the ones you need by uncommenting them in your `opencode.json`:

| Server name | Module | Tools |
|---|---|---|
| `chemgraph` | `chemgraph.mcp.mcp_tools` | molecule_name_to_smiles, smiles_to_coordinate_file, run_ase, extract_output_json |
| `chemgraph-mace-parsl` | `chemgraph.mcp.mace_mcp_parsl` | MACE ensemble calculations via Parsl (HPC) |
| `chemgraph-graspa-parsl` | `chemgraph.mcp.graspa_mcp_parsl` | gRASPA gas adsorption via Parsl (HPC) |
| `chemgraph-data-analysis` | `chemgraph.mcp.data_analysis_mcp` | CIF splitting, JSONL aggregation, isotherm plotting |

### How it works

OpenCode spawns the MCP server as a local child process using stdio transport. The `{env:CHEMGRAPH_PYTHON}` variable in the config is resolved at startup, so different users (or the same user on different machines) can each point to their own ChemGraph installation without modifying the committed config.

## Notes for Parsl-based servers

`mace_mcp_parsl.py` and `graspa_mcp_parsl.py` rely on Parsl and HPC-specific configuration. Ensure your environment is prepared for the target system before running production jobs.
