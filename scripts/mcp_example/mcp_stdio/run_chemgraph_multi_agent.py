"""Local multi-agent MCP example over stdio.

Spawns ChemGraph's built-in MCP server (chemgraph.mcp.mcp_tools) as a
local subprocess, loads the exposed tools, and runs ChemGraph with the
multi_agent workflow.  The planner decomposes the query into parallel
executor tasks (one per molecule) using the Send() pattern.

Usage
-----
    cd scripts/mcp_example/mcp_stdio
    python run_chemgraph_multi_agent.py

Prerequisites
-------------
- ChemGraph installed (``pip install -e .`` from the repo root)
- ``OPENAI_API_KEY`` environment variable set
"""

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# A prompt that naturally decomposes into two parallel subtasks,
# showcasing the planner/executor fan-out.
prompt = (
    "Compare the enthalpy of water and CO2 using MACE at 300K. "
    "Report the values and which molecule has the higher enthalpy."
)

# ---------------------------------------------------------------------------
# MCP client — local stdio transport (no SSH, no remote node)
# Uses ChemGraph's built-in MCP server module (no local copy needed)
# ---------------------------------------------------------------------------

client = MultiServerMCPClient(
    {
        "ChemGraph General Tools": {
            "command": sys.executable,  # current Python interpreter
            "args": ["-m", "chemgraph.mcp.mcp_tools"],
            "transport": "stdio",
        },
    }
)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


async def bootstrap():
    # 1. Load MCP tools from the local server
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}")

    # 2. Create ChemGraph with multi-agent workflow (MCP tools work natively)
    cg = ChemGraph(
        model_name="gpt-4o",
        workflow_type="multi_agent",
        structured_output=False,
        return_option="state",
        tools=tools,
    )

    # 3. Run the query
    result = await cg.run(prompt)

    # 4. Print final state summary
    print("\n######## FINAL STATE #########")
    if isinstance(result, dict) and "messages" in result:
        for msg in result["messages"]:
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            # Truncate long tool outputs for readability
            if len(str(content)) > 500:
                content = str(content)[:500] + "..."
            print(f"[{role}] {content}")
    else:
        print(result)


if __name__ == "__main__":
    asyncio.run(bootstrap())
