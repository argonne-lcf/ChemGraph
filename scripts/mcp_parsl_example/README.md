# Using MCP + Parsl on Aurora or Polaris

This example shows how to run a ChemGraph Parsl-backed MCP server on an HPC
compute node and connect a ChemGraph client to it through port forwarding.

Use this pattern when the LLM client should stay on a login node or workstation,
while simulation tools run on allocated compute resources.

## Prerequisites

- ChemGraph installed in the Python environment used on the compute node.
- `parsl` installed, either through the optional dependency or directly:

  ```bash
  pip install -e ".[parsl]"
  ```

- Any workflow-specific runtime installed, such as gRASPA, MACE, or FDMNES.
- Provider credentials for the LLM client, such as `OPENAI_API_KEY` or the
  endpoint-specific variables used by your deployment.

## 1. Request a compute node

For a PBS-based ALCF system:

```bash
qsub -I -q debug -l select=1,walltime=01:00:00 -A your_account -l filesystems=home:flare
```

After the allocation starts, connect to the compute node if your site requires a
separate SSH hop:

```bash
ssh YOUR_COMPUTE_NODE_ID
```

## 2. Start the Parsl-backed MCP server

Activate your environment and select the target system:

```bash
module load frameworks
source /path/to/venv/bin/activate

export COMPUTE_SYSTEM=aurora  # or polaris
export CHEMGRAPH_LOG_DIR="$PWD/chemgraph_mcp_logs"
export http_proxy="proxy.alcf.anl.gov:3128"
export https_proxy="proxy.alcf.anl.gov:3128"
export NO_PROXY=127.0.0.1,localhost,::1
```

Start one of the Parsl-backed MCP servers. For gRASPA:

```bash
python -m chemgraph.mcp.graspa_mcp_parsl \
  --transport streamable_http \
  --host 0.0.0.0 \
  --port 9001
```

For XANES/FDMNES:

```bash
python -m chemgraph.mcp.xanes_mcp_parsl \
  --transport streamable_http \
  --host 0.0.0.0 \
  --port 9007
```

`mace_mcp_parsl.py` is also available, but it contains site-specific
`worker_init` settings in the module. Review module loads, conda environment
paths, and filesystem paths before using it in production.

## 3. Or submit the server as a batch job

Edit `start_mcp_server.sub` and update:

- `#PBS -A your_account`
- the environment activation path
- `COMPUTE_SYSTEM`
- the MCP module and port if you want a server other than gRASPA

Then submit:

```bash
qsub start_mcp_server.sub
```

Find the compute node assigned to the job:

```bash
qstat -f JOB_ID | awk -F'=' '/exec_host =/ {gsub(/^[ \t]+/,"",$2); sub(/\/.*/,"",$2); print $2}'
```

## 4. Forward the MCP port

From the login node, forward the server port from the compute node:

```bash
ssh -N -L 9001:localhost:9001 YOUR_COMPUTE_NODE_ID
```

Keep this terminal open while the client runs.

## 5. Run the ChemGraph client

In another terminal on the login node:

```bash
module load frameworks
source /path/to/venv/bin/activate
export NO_PROXY=127.0.0.1,localhost,::1
export no_proxy=127.0.0.1,localhost,::1

python run_mcp_parsl.py
```

The example client connects to `http://127.0.0.1:9001/mcp/`.

## Troubleshooting

- If the client gets `503 Service Unavailable`, verify that the MCP server is
  still running and that the SSH tunnel points to the correct compute node.
- If localhost requests go through the site proxy, set both `NO_PROXY` and
  `no_proxy` for `127.0.0.1,localhost,::1`.
- If Parsl fails at startup, confirm that `PBS_NODEFILE` exists inside the
  allocation and that `COMPUTE_SYSTEM` matches a supported config.
