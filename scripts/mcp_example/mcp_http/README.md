# Using MCP via Port-Forwarding on Aurora (ALCF)

This directory provides an example of how to use **MCP (Model Control Protocol)** with port-forwarding on **Aurora at ALCF**. The instructions below guide you through launching the MCP server and connecting ChemGraph to it.

## Prerequisites

- ChemGraph installed in your environment
- `OPENAI_API_KEY` set (or enter interactively when running ChemGraph)

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
Navigate to this directory, activate the environment and start the MCP server:
```bash
# Set proxy for tools that query external databases 
export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"

# Load environment modules and activate your Python environment
module load frameworks
source /path/to/venv/bin/activate

# Start MCP server
python start_mcp_server.py
```
The server will run on port 9001 by default.

You can also launch the MCP server as a batch job using the `start_mcp_server.sub` script.
First, open the script and update the placeholders for your account name and path to your virtual environment.
Then submit the job with:
```bash
qsub start_mcp_server.sub
```
Once the job is running, you can find the compute node ID with:
```bash
qstat -f <JOB_ID> | awk -F'=' '/exec_host =/ {gsub(/^[ \t]+/,"",$2); sub(/\/.*/,"",$2); print $2}'
```

### 4. Set Up Port Forwarding
Open a new terminal on the login node, forwarding port 9001 so you can access the MCP server running on the compute node:
```bash
ssh -N -L 9001:localhost:9001 YOUR_COMPUTE_NODE_ID
```
Keep this terminal open while using ChemGraph. This ensures that all traffic from Aurora compute node to login node is routed through port 9001.

### 5. Launch ChemGraph
In another terminal session on the same login node used in Step 4, run ChemGraph and connect it to the MCP server (listening on port 9001 by default):
```bash
# Load environment modules and activate your Python environment
module load frameworks
source /path/to/venv/bin/activate

python run_chemgraph.py
```

### Troubleshooting

If you get an error like this:
```
httpx.HTTPStatusError: Server error '503 Service Unavailable' for url 'http://127.0.0.1:9001/mcp/'
```
Try:
```
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
```
And run ChemGraph again.
=======
Finally, in another terminal, activate the environment and run ChemGraph to connect to the MCP server, which listens on port 9001.
```bash
python scripts/mcp_example/mcp_http/run_mcp.py
```
python run_chemgraph.py
```