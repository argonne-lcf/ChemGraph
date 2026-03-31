"""
Run XANES workflows via the ChemGraph agent using MCP HTTP transport.

Connects to an already-running XANES MCP server via HTTP.  The LLM agent
receives the XANES MCP tools and uses them to fulfill the natural language
prompt.

Prerequisites:
  - XANES MCP server running (via start_mcp_server.sub or manually)
  - SSH tunnel set up if server is on a compute node
  - OPENAI_API_KEY set in environment (or another LLM provider key)
  - MP_API_KEY set on the server side (for prompts that fetch from Materials Project)

Usage:
  export OPENAI_API_KEY="your_key"
  python run_chemgraph.py
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_NAME = "gpt-4o-mini"
MCP_URL = "http://127.0.0.1:9007/mcp/"

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
    "(Z_absorber=26) with a cluster radius of 6.0 Angstrom."
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
            "transport": "streamable_http",
            "url": MCP_URL,
        },
    }
)


async def main():
    tools = await client.get_tools()
    print(f"Connected to XANES MCP server at {MCP_URL}")
    print(f"Available tools: {[t.name for t in tools]}")
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}\n")

    cg = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type="single_agent_xanes",
        structured_output=False,
        return_option="state",
        tools=tools,
    )

    result = await cg.run(PROMPT)
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
