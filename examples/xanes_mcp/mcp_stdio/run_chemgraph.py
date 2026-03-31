"""
Run XANES workflows via the ChemGraph agent using MCP stdio transport.

The MCP server is launched locally as a subprocess -- no separate server
process, SSH tunnel, or port forwarding needed.  The LLM agent receives
the XANES MCP tools and uses them to fulfill the natural language prompt.

Prerequisites:
  - OPENAI_API_KEY set in environment (or another LLM provider key)
  - FDMNES_EXE set in environment
  - MP_API_KEY set in environment (for prompts that fetch from Materials Project)

Usage:
  export OPENAI_API_KEY="your_key"
  export MP_API_KEY="your_mp_key"
  export FDMNES_EXE="/path/to/fdmnes"

  python run_chemgraph.py
"""

import asyncio
import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_NAME = "gpt4o"
MCP_SERVER_MODULE = "chemgraph.mcp.xanes_mcp"

# ==============================================================================
# EXAMPLE PROMPTS
#
# Uncomment one prompt at a time, or set PROMPT to your own query.
# ==============================================================================

# --- Single structure XANES ---
# PROMPT = (
#     "Run a XANES calculation on the file /path/to/Fe2O3.cif "
#     "at the Fe K-edge (Z_absorber=26) with a cluster radius of 6.0 Angstrom."
# )

# --- Fetch + single XANES ---
PROMPT = (
    "Fetch optimized structures for Fe2O3 from Materials Project, "
    "then run XANES calculations on each structure at the Fe K-edge "
    "(Z_absorber=26) with a cluster radius of 3.0 Angstrom."
)

# --- Fetch + XANES + plot ---
# PROMPT = (
#     "Fetch optimized structures for CoO from Materials Project, "
#     "run XANES calculations on each structure at the Co K-edge "
#     "(Z_absorber=27) with a cluster radius of 5.0 Angstrom, "
#     "and then generate normalized XANES plots for the results."
# )

# --- Multiple systems ---
# PROMPT = (
#     "Fetch structures for NiO and FeO from Materials Project, "
#     "then run XANES calculations on each structure separately. "
#     "Use Z_absorber=28 for NiO (Ni K-edge) and Z_absorber=26 for FeO (Fe K-edge). "
#     "Use a cluster radius of 6.0 Angstrom for all calculations."
# )

# ==============================================================================


client = MultiServerMCPClient(
    {
        "XANES MCP": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-u", "-m", MCP_SERVER_MODULE],
            "env": {**os.environ},
        },
    }
)


async def main():
    tools = await client.get_tools()
    print(f"Connected to XANES MCP server via stdio (local subprocess)")
    print(f"Available tools: {[t.name for t in tools]}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}\n")

    cg = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type="single_agent_xanes",
        structured_output=False,
        return_option="state",
        tools=tools,
        argo_user="tpham",
        base_url="https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/",
    )

    result = await cg.run(PROMPT)
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
