import asyncio, sys, json
from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent_mcp import ChemGraph
from chemgraph.tools.ase_tools import run_ase

prompt_single = "What is the enthalpy of CO2 using TBLite GFN2-xTB at 400K?"
prompt_multi = "You are given a chemical reaction: 1 (Nitrogen gas) + 3 (Hydrogen gas) -> 2 (Ammonia). Calculate the enthalpy change (detal H) for this reaction using GFN2-xTB at 400K."

client = MultiServerMCPClient({
    "Chemistry Tools MCP": {
        "command": "python",
        "args": ["/Users/tpham2/work/projects/ChemGraph/mcp_tools.py"],
        "transport": "stdio",
    },
})


async def bootstrap():
    tools = await client.get_tools()
    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        # workflow_type="multi_agent",
        structured_output=True,
        return_option="state",
        tools=tools,
    )
    result = await cg.run(prompt_single)
    print(result)


asyncio.run(bootstrap())
