# Integrating ChemGraph with OpenCode on ALCF Machines

This guide walks through setting up [OpenCode](https://opencode.ai/) to use
[ChemGraph](https://github.com/argonne-lcf/ChemGraph) as an MCP (Model Context
Protocol) tool server on ALCF systems such as Aurora. With this setup, you can
use natural-language prompts inside OpenCode to run computational chemistry
workflows (geometry optimizations, vibrational analyses, thermochemistry
calculations and more) on ALCF compute nodes.

## Architecture Overview

Everything runs from an **Aurora login node**. Two SSH tunnels provide
connectivity to the Argo LLM API and the ChemGraph MCP server on a compute node:

```
Argo API                  Aurora Login Node            Compute Node
(apps-dev.inside.anl.gov) (you are here)
┌───────────────┐          ┌────────────┐              ┌───────────────┐
│  Argo LLM     │◄── SSH ──┤            ├──── SSH ────►│  ChemGraph    │
│  Gateway      │  Tunnel  │  OpenCode  │    Tunnel    │  MCP Server   │
│  :443         │  (8443)  │            │    (9003)    │  (port 9003)  │
└───────────────┘          └────────────┘              └───────────────┘
```

| Tunnel | Purpose | Endpoint |
|--------|---------|----------|
| Port **8443** | Argo LLM API gateway | `apps-dev.inside.anl.gov:443` |
| Port **9003** | ChemGraph MCP server | Compute node MCP process |

## Prerequisites

1. **ChemGraph** installed on ALCF — see
   [`scripts/mcp_example/installation.md`](../../scripts/mcp_example/installation.md)
   for Aurora-specific instructions.
2. **OpenCode** installed on the Aurora login node — see
   <https://opencode.ai/docs/installation>.
3. **SSH access** from the Aurora login node to an Argonne machine with Argo API
   access (for the Argo tunnel).
4. **Allocation** on an ALCF system (Aurora, Polaris, etc.) to request compute
   nodes.

---

## Step 1: Setting Up OpenCode on ALCF Machines

On the **Aurora login node**, create (or update) the OpenCode **user
configuration** file at `~/.config/opencode/opencode.json`. This tells OpenCode
how to reach the Argo LLM API:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "argo": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Argo",
      "options": {
        "baseURL": "https://127.0.0.1:8443/argoapi/v1",
        "apiKey": "<ANL Username>",
        "headers": {
          "Authorization": "Bearer custom-token",
          "Host": "apps-dev.inside.anl.gov"
        }
      },
      "models": {
        "gpt52": {
          "name": "GPT-5.2"
        },
        "claudeopus46": {
          "name": "Claude Opus 4.6",
        },
        "claudesonnet46": {
          "name": "Claude Sonnet 4.6",
        },
        "claudesonnet45": {
          "name": "Claude Sonnet 4.5",
        },
        "claudehaiku45": {
          "name": "Claude Haiku 4.5",
        }
      }
    }
  },
  "lsp": {
    "pyright": {
      "command": ["/home/<ANL Username>/.local/bin/pyright-langserver", "--stdio"],
      "extensions": [".py", ".pyi"]
    }
  }
}
```

**Key fields:**

| Field | Description |
|-------|-------------|
| `baseURL` | Points to `127.0.0.1:8443` on the login node — the local end of the SSH tunnel to Argo (see Step 2). The port must match your SSH `-L` port. |
| `apiKey` | Your ANL username. |
| `Host` header | Routes traffic to the correct backend (`apps-dev.inside.anl.gov`). |
| `models` | Available LLMs via Argo. |
| `lsp.pyright` | Required. Enables Python language server support in OpenCode. Update the path to match your `pyright-langserver` location. |

---

## Step 2: Connect OpenCode to Argo

From the **Aurora login node**, open a terminal and create an SSH tunnel to the
Argo API gateway through an Argonne machine that has Argo access:

```bash
ssh -L 8443:apps-dev.inside.anl.gov:443 <your_argonne_machine_with_Argo_access> -N
```

- `-L 8443:apps-dev.inside.anl.gov:443` forwards login-node port `8443` to the
  Argo endpoint.
- `-N` keeps the connection open without starting a shell.
- This tunnel must remain active while you use OpenCode.

> **Note:** The port number (`8443`) must match the port in the `baseURL` field
> of your `~/.config/opencode/opencode.json`. You can choose any available port
> on the login node — just keep them consistent.

---

## Step 3: Start the ChemGraph MCP Server

The MCP server runs on an ALCF **compute node** so it has access to GPUs and the
software stack needed for molecular simulations.

### 3a. Request an interactive compute node

From the **Aurora login node**, request a compute node:

```bash
qsub -I -l select=1 -l walltime=01:00:00 -l filesystems=home:flare -q debug -A <your_project>
```

### 3b. Start the MCP server on the compute node

A convenience script is provided in this directory:

```bash
./start_mcp_interactive.sh --venv /path/to/your/chemgraph/venv --port 9003
```

The script will:

1. Set ALCF proxy variables and load the `frameworks` module.
2. Activate the specified virtual environment.
3. Start the ChemGraph MCP HTTP server on the given port.
4. Wait for the server to become ready and print connection instructions.
5. Tail the server log until you press `Ctrl+C`.

**Script options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--port PORT` | `9003` | Port for the MCP HTTP server |
| `--venv PATH` | _(none)_ | Path to the Python virtual environment |
| `--log-dir PATH` | `./chemgraph_mcp_logs` | Directory for server logs |
| `--mcp-module MOD` | `chemgraph.mcp.mcp_tools` | Python module to run as the MCP server |

Take note of the compute node hostname as you will need
it for the next step.

---

## Step 4: Connect OpenCode to the MCP Server

### 4a. Set up an SSH tunnel to the compute node

From a **second terminal on the Aurora login node**, tunnel port 9003 to the
compute node:

```bash
ssh -N -L 9003:localhost:9003 <compute_node_id>
```

Replace `<compute_node_id>` with the hostname from Step 3.
 This forwards login-node port 9003 to the MCP server running
on the compute node.

### 4b. Place the MCP configuration file

Copy the provided [`opencode.jsonc`](./opencode.jsonc) into your **ChemGraph
project working directory** (the directory where you will run `opencode`):

```bash
cp examples/OpenCode/opencode.jsonc /path/to/your/working/directory/opencode.json
```

Or, if you are already in the ChemGraph root:

```bash
cp examples/OpenCode/opencode.jsonc ./opencode.json
```

The MCP config tells OpenCode where to find the ChemGraph MCP server:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "chemgraph": {
      "type": "remote",
      "url": "http://localhost:9003/mcp/",
      "enabled": true,
      "headers": {
        "Authorization": "Bearer MY_API_KEY"
      }
    }
  }
}
```

> **Note:** The `url` port (`9003`) must match the port used in the SSH tunnel
> and the `--port` argument to `start_mcp_interactive.sh`. The project root
> already contains an `opencode.json` for **local** (stdio) MCP usage. The
> config in this example is for **remote** (HTTP) MCP via port forwarding.

---

## Step 5: Launch OpenCode

With both SSH tunnels active on the login node (Argo on port 8443, MCP on port
9003), open a **third terminal on the Aurora login node** and start OpenCode in
your working directory:

```bash
opencode
```

OpenCode will:

1. Load the user config from `~/.config/opencode/opencode.json` (Argo provider).
2. Detect the project-level `opencode.json` and connect to the ChemGraph MCP
   server.
3. Display available MCP tools (e.g., `molecule_name_to_smiles`, `run_ase`,
   `smiles_to_coordinate_file`, `extract_output_json`).

You can verify the MCP connection by pressing `ctrl+p` and checking that the
ChemGraph tools are listed.

---

## Example Queries

Once connected, try these prompts inside OpenCode:

```
What is the enthalpy of CO2 using MACE at 500K?
```

```
Optimize the geometry of aspirin using MACE-MP medium model.
```

```
Calculate the vibrational frequencies of water using TBLite GFN2-xTB.
```
---

## Summary of Connections

All terminals below are on the **Aurora login node**:

| Terminal | Command | Purpose |
|----------|---------|---------|
| 1 | `ssh -L 8443:apps-dev.inside.anl.gov:443 <argo_host> -N` | Tunnel to Argo LLM API |
| 2 | `qsub -I ...` then `./start_mcp_interactive.sh ...` | Start MCP server on compute node |
| 3 | `ssh -N -L 9003:localhost:9003 <compute_node>` | Tunnel to MCP server |
| 4 | `opencode` | Launch OpenCode |

---

## Troubleshooting

### 503 Service Unavailable or proxy errors

If you encounter 503 errors or proxy-related failures, unset the proxy
environment variables on the login node before running OpenCode:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
```

### MCP server fails to start

- Check the log file printed by `start_mcp_interactive.sh` (in
  `./chemgraph_mcp_logs/`).
- Ensure ChemGraph is installed correctly: `python -c "import chemgraph"`.
- Verify the virtual environment path passed to `--venv` is correct.

### Cannot connect to MCP server

- Confirm the SSH tunnel (port 9003) is active on the login node.
- Verify the compute node hostname matches what you used in the SSH tunnel.
- Test the endpoint from the login node: `curl http://localhost:9003/mcp/`.

### Argo connection issues

- Confirm the SSH tunnel (port 8443) is active.
- Verify your ANL username is set as the `apiKey` in the OpenCode config.

---

## Acknowledgements

- **Dr. Neil Getty (ANL)** for the config for OpenCode on ALCF machines to Argo API.
