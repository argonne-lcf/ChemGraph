"""Single-agent MCP example over HTTP.

Connects to ChemGraph's built-in MCP server running on a remote ALCF
compute node via HTTP (streamable_http transport with port forwarding).

Usage
-----
    # 1. Start the MCP server on a compute node (see README.md)
    # 2. Set up SSH port forwarding to the compute node
    # 3. Run from a login node:
    python run_chemgraph.py

The MCP server should be started with:
    python -m chemgraph.mcp.mcp_tools --transport streamable_http --port 9003
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MCP_PORT = 9003  # Must match the port used when starting the MCP server

prompt_single = "What is the enthalpy of CO2 using MACE medium at 500K?"

client = MultiServerMCPClient({
    "ChemGraph General Tools": {
        "transport": "streamable_http",
        "url": f"http://127.0.0.1:{MCP_PORT}/mcp/",
    },
})


async def bootstrap():
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        structured_output=False,
        return_option="state",
        tools=tools,
    )
    result = await cg.run(prompt_single)
    print(result)


asyncio.run(bootstrap())
