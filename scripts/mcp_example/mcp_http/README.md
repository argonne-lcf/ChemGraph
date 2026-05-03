# Using MCP via Port-Forwarding on Aurora (ALCF)

This directory provides an example of how to use **MCP (Model Context Protocol)** with port-forwarding on **Aurora at ALCF**. All scripts use ChemGraph's built-in MCP server module (`chemgraph.mcp.mcp_tools`) — no local copy of the server code is needed.

## Prerequisites

- ChemGraph installed in your environment
- `OPENAI_API_KEY` set (or enter interactively when running ChemGraph)

## Files

| File | Description |
|---|---|
| `run_chemgraph.py` | Client script — connects to the MCP server and runs a query |
| `start_mcp_server.py` | Convenience wrapper to start the MCP server via HTTP |
| `start_mcp_server.sub` | PBS batch script (simple) |
| `start_mcp_server_http.sub` | PBS batch script (with logging and connection info) |
| `start_mcp_server_interactive.sh` | Shell script for interactive compute sessions |

## Step-by-Step Instructions

### 1. Secure a Compute Node

Request an interactive job on a compute node:

```bash
qsub -I -q debug -l select=1,walltime=60:00 -A your_account_name -l filesystems=flare
```

### 2. SSH to the Compute Node
```bash
ssh YOUR_COMPUTE_NODE_ID
```

### 3. Launch the MCP Server
Activate the environment and start ChemGraph's built-in MCP server:
```bash
# Set proxy for tools that query external databases
export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"

# Load environment modules and activate your Python environment
module load frameworks
source /path/to/venv/bin/activate

# Start MCP server on port 9003 (using ChemGraph's built-in module)
python -m chemgraph.mcp.mcp_tools --transport streamable_http --port 9003

# Or use the convenience wrapper:
# python start_mcp_server.py
```

You can also launch the MCP server as a batch job using `start_mcp_server.sub` or `start_mcp_server_http.sub`.
First, open the script and update the placeholders for your account name and path to your virtual environment.
Then submit the job with:
```bash
qsub start_mcp_server_http.sub
```
Once the job is running, you can find the compute node ID with:
```bash
qstat -f <JOB_ID> | awk -F'=' '/exec_host =/ {gsub(/^[ \t]+/,"",$2); sub(/\/.*/,"",$2); print $2}'
```

### 4. Set Up Port Forwarding
Open a new terminal on the login node, forwarding port 9003 so you can access the MCP server running on the compute node:
```bash
ssh -N -L 9003:localhost:9003 YOUR_COMPUTE_NODE_ID
```
Keep this terminal open while using ChemGraph.

### 5. Launch ChemGraph
In another terminal session on the same login node used in Step 4, run ChemGraph and connect it to the MCP server:
```bash
# Load environment modules and activate your Python environment
module load frameworks
source /path/to/venv/bin/activate

python run_chemgraph.py
```

### Troubleshooting

If you get an error like this:
```
httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'http://127.0.0.1:9003/mcp/'
```
Try:
```
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```
And run ChemGraph again.