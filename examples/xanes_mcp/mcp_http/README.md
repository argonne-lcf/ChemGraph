# XANES via MCP HTTP (Port Forwarding)

Run XANES workflows using the ChemGraph LLM agent connected to a running XANES MCP server via HTTP transport.

## Prerequisites

- ChemGraph installed in your environment
- `OPENAI_API_KEY` set (or another LLM provider key)
- `FDMNES_EXE` set (path to the FDMNES executable, on the server side)
- `MP_API_KEY` set (for prompts that fetch from Materials Project)

## Files

| File | Description |
|------|-------------|
| `run_chemgraph.py` | LLM agent client with example prompts |
| `start_mcp_server.py` | Start the XANES MCP server (simple Python launcher) |
| `start_mcp_server.sub` | PBS batch script to launch the server as a job |
| `start_mcp_server_interactive.sh` | Shell script for interactive sessions |

## Step-by-Step

### 1. Start the MCP Server

**Option A: Interactive script (recommended)**

```bash
export FDMNES_EXE="/path/to/fdmnes"
export MP_API_KEY="your_mp_key"

./start_mcp_server_interactive.sh --venv /path/to/venv --port 9007
```

**Option B: Python launcher**

```bash
source /path/to/venv/bin/activate
export FDMNES_EXE="/path/to/fdmnes"

python start_mcp_server.py --port 9007
```

**Option C: PBS batch job (HPC)**

Edit `start_mcp_server.sub` and update `VENV_PATH`, `FDMNES_EXE_PATH`, and `MP_API_KEY_VALUE`, then:

```bash
qsub start_mcp_server.sub
```

Find the compute node:

```bash
cat chemgraph_xanes_logs/connection_info.txt
```

### 2. Set Up Port Forwarding (if remote)

If the server is on a remote compute node, forward port 9007 from the login node:

```bash
ssh -N -L 9007:localhost:9007 COMPUTE_NODE
```

Keep this terminal open.

### 3. Run ChemGraph

In another terminal:

```bash
source /path/to/venv/bin/activate

export OPENAI_API_KEY="your_key"
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1

python run_chemgraph.py
```

## Example Prompts

The script includes several example prompts (uncomment one at a time in `run_chemgraph.py`):

| Prompt | What it does |
|--------|-------------|
| Fetch + single XANES (default) | Fetches Fe2O3 from Materials Project, runs XANES on each structure |
| Single structure XANES | Runs XANES on a provided CIF file directly |
| Fetch + XANES + plot | Fetches CoO, runs XANES, generates normalized plots |
| Multiple systems | Fetches NiO and FeO, runs XANES on each structure |

## Configuration

Edit `run_chemgraph.py` to change:

- `MODEL_NAME` -- the LLM model to use (default: `gpt-4o-mini`)
- `MCP_URL` -- the MCP server URL (default: `http://127.0.0.1:9007/mcp/`)
- `PROMPT` -- uncomment a different example prompt or write your own

## Troubleshooting

If you get `503 Service Unavailable`, set the proxy bypass variables:

```bash
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1
```
