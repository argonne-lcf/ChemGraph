# Using MCP via Port-Forwarding on Aurora (ALCF)

This directory provides an example of how to use **MCP (Model Control Protocol)** with port-forwarding on **Aurora HPC at ALCF**. The instructions below guide you through launching the MCP server and connecting ChemGraph to it.

## Prerequisites

- ChemGraph installed in your environment  

## Step-by-Step Instructions

### 1. Secure a Compute Node

Request an interactive job on a compute node:

```bash
qsub -I -q debug -l select=1,walltime=60:00 -A your_account_name -l filesystems=flare
```
### 2. SSH to the Compute Node
```bash
ssh YOUR_LOGIN_NODE_ID
```
### 3. Launch the MCP Server
Navigate to this directory and start the MCP server:
```bash
# Set proxy for tools that query external databases 
export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"

# Load environment modules and activate your Python environment
module load frameworks
source /path/to/venv/bin/activate

# Start MCP server
python mcp_tools_http.py
```
The server will run on port 9001 by default.
### 4. Set Up Port Forwarding
Open a new terminal on the login node, forwarding port 9001 so you can access the MCP server running on the compute node:
```bash
ssh -N -L 9001:localhost:9001 YOUR_LOGIN_NODE_ID
```
Keep this terminal open while using ChemGraph. This ensures that all traffic to localhost:9001 on your local machine is securely routed to the MCP server on Aurora.

### 5. Launch ChemGraph
In another terminal session on the same login node used in Step 4, run ChemGraph and connect it to the MCP server (listening on port 9001 by default):
```bash
# Load environment modules and activate your Python environment
module load frameworks
source /path/to/venv/bin/activate

python run_chemgraph.py
```
