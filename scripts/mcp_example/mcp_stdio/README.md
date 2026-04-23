# Using MCP via stdio

This directory provides examples of how to use **MCP (Model Context Protocol)** with stdio transport. Two scripts are included:

| Script | Workflow | Description |
|---|---|---|
| `run_chemgraph.py` | `single_agent` | Single agent via MCP over SSH to an ALCF compute node |
| `run_chemgraph_multi_agent.py` | `multi_agent` | **Multi-agent** (Planner/Executor) via MCP running **locally** |

Both scripts use ChemGraph's built-in MCP server (`chemgraph.mcp.mcp_tools`) — no local copy of the server code is needed.

## Prerequisites

- ChemGraph installed (`pip install -e .` from the repo root)
- `OPENAI_API_KEY` set

---

## Local Multi-Agent Example (`run_chemgraph_multi_agent.py`)

The multi-agent example runs entirely on your local machine. The
`MultiServerMCPClient` spawns `python -m chemgraph.mcp.mcp_tools` as a
child process over stdio — no SSH or remote nodes needed.

### How it works

1. The MCP client spawns ChemGraph's built-in MCP server locally via stdio.
2. ChemGraph creates a **Planner/Executor** workflow (`multi_agent`).
3. The **Planner** decomposes the query into parallel subtasks (e.g., one
   per molecule).
4. Each **Executor** runs independently with access to MCP tools, using
   the `Send()` fan-out pattern.
5. Results are aggregated back to the Planner for a final answer.

### Run it

```bash
cd scripts/mcp_example/mcp_stdio
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python run_chemgraph_multi_agent.py
```

### Customization

Edit the script to change:
- `model_name` — any supported LLM (default: `gpt-4o`)
- `prompt` — the chemistry query to decompose
- `structured_output` — set to `True` to get a formatted `ResponseFormatter` JSON

---

## Remote Single-Agent Example (`run_chemgraph.py`)

This example connects to ChemGraph's MCP server running on an **ALCF Aurora** compute node via SSH. The SSH command runs `python -m chemgraph.mcp.mcp_tools` on the remote node.

### 1. Secure a Compute Node

```bash
qsub -I -q debug -l select=1,walltime=60:00 -A your_account_name -l filesystems=flare
```

### 2. Edit variables in `run_chemgraph.py`

```python
REMOTE_HOST = "YOUR_COMPUTE_NODE_ID"
REMOTE_ENV = "path/to/venv"
```

### 3. Launch from a login node

```bash
module load frameworks
source /path/to/venv/bin/activate
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
python run_chemgraph.py
```
