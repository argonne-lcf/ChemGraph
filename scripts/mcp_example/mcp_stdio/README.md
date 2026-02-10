# Using MCP via stdio on Aurora (ALCF)

This directory provides an example of how to use **MCP (Model Control Protocol)** with stdio on **Aurora at ALCF**. The instructions below guide you through launching the MCP server and connecting ChemGraph to it.

## Prerequisites

- ChemGraph installed in your environment 
- `OPENAI_API_KEY` set (or enter interactively when running ChemGraph)

## Step-by-Step Instructions

### 1. Secure a Compute Node

Request an interactive job on a compute node:

```bash
qsub -I -q debug -l select=1,walltime=60:00 -A your_account_name -l filesystems=flare
```
### 2. Edit variables in run_chemgraph.py
Update the following variables to match your setup:
```
REMOTE_HOST = "YOUR_COMPUTE_NODE_ID"
REMOTE_ENV = "path/to/venv"
```
### 3. Launch MCP server and ChemGraph on a login node
```bash
# Load the environments
module load frameworks
source /path/to/venv/bin/activate

# Set your API key for the OpenAI model (GPT-4o-mini in this example)
export OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

# Start ChemGraph
python run_chemgraph.py
```
