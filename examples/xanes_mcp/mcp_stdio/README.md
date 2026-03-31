# XANES via MCP stdio (Local Subprocess)

Run XANES workflows using the ChemGraph LLM agent with the XANES MCP server launched locally as a subprocess via stdio transport. No separate server process, SSH tunnel, or port forwarding needed.

## Prerequisites

- ChemGraph installed in your environment
- `OPENAI_API_KEY` set (or another LLM provider key)
- `MP_API_KEY` set (for prompts that fetch from Materials Project)
- `FDMNES_EXE` set (path to the FDMNES executable)

## Usage

```bash
# Set environment variables
export OPENAI_API_KEY="your_key"
export MP_API_KEY="your_mp_key"
export FDMNES_EXE="/path/to/fdmnes"

# Run with the default prompt (fetch Fe2O3 + run XANES)
python run_chemgraph.py
```

## Example Prompts

The script includes several example prompts (uncomment one at a time):

| Prompt | What it does |
|--------|-------------|
| Fetch + single XANES (default) | Fetches Fe2O3 from Materials Project, runs XANES on each structure |
| Single structure XANES | Runs XANES on a provided CIF file directly |
| Fetch + XANES + plot | Fetches CoO, runs XANES, generates normalized plots |
| Multiple systems | Fetches NiO and FeO, runs XANES on each structure |

## How It Works

1. The script launches `chemgraph.mcp.xanes_mcp` as a local subprocess using stdio transport
2. The MCP client discovers the available tools (`fetch_mp_structures`, `run_xanes_single`, `plot_xanes`)
3. A `ChemGraph` agent is created with the `single_agent_xanes` workflow
4. The LLM receives the prompt and autonomously calls the appropriate tools to complete the task
