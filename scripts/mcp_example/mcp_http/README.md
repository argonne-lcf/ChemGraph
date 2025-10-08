# Using MCP via Port-Forwarding on Aurora (ALCF)

This directory provides an example of how to use **MCP (Model Control Protocol)** with port-forwarding on **Aurora HPC at ALCF**. The instructions below guide you through launching the MCP server and connecting ChemGraph to it.

## Prerequisites

- ChemGraph installed in your environment  
- Access to Aurora HPC at ALCF  
- A valid ALCF account

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
python mcp_tools_http.py
```
The server will run on port 9001 by default.
### 4. Set Up Port Forwarding
In a separate terminal, establish port forwarding to access the MCP server locally:
```bash
ssh -L 9001:localhost:9001 YOUR_LOGIN_NODE_ID
```
### 5. Launch ChemGraph
Finally, in another terminal, run ChemGraph to connect to the MCP server, which listens on port 9001.
```bash
python run_mcp.py
```
