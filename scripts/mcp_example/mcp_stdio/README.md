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
### 2. Edit the following variables in run_mcp.py file
```
REMOTE_HOST = "YOUR_COMPUTE_NODE" # Your compute node ID from step 1
CONDA_ENV = "YOUR_CONDA_ENV" # Environment with ChemGraph installed
MCP_SERVER = "PATH/TO/mcp_tools_stdio.py"
```
### 3. Launch both MCP server and ChemGraph
```bash
python run_mcp.py
```
