"""Single-agent MCP example over stdio (remote via SSH).

Connects to ChemGraph's built-in MCP server running on a remote ALCF
compute node via SSH + stdio transport.

Usage
-----
    # 1. Secure a compute node (qsub -I ...)
    # 2. Edit REMOTE_HOST and REMOTE_ENV below
    # 3. Run from a login node:
    python run_chemgraph.py
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

# ---------------------------------------------------------------------------
# Configuration -- edit these to match your setup
# ---------------------------------------------------------------------------
REMOTE_HOST = "YOUR_COMPUTE_NODE"  # e.g. "x4703c4s5b0n0"
REMOTE_ENV = "/path/to/venv"       # e.g. "/lus/flare/projects/.../venv"

# Uses ChemGraph's built-in MCP server module (no local copy needed)
REMOTE_CMD = (
    f"module load frameworks && source {REMOTE_ENV}/bin/activate && "
    f"export http_proxy='proxy.alcf.anl.gov:3128' && "
    f"export https_proxy='proxy.alcf.anl.gov:3128' && "
    f"python -u -m chemgraph.mcp.mcp_tools"
)

prompt_single = "What is the enthalpy of CO2 using TBLite GFN2-xTB at 400K?"

client = MultiServerMCPClient({
    "ChemGraph General Tools": {
        "command": "ssh",
        "args": [
            REMOTE_HOST,
            REMOTE_CMD,
        ],
        "transport": "stdio",
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
    await cg.run(prompt_single)


asyncio.run(bootstrap())
