import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

# prompt_single = "What is the enthalpy of water using TBLite GFN2-xTB at 400K?"
prompt_single = "What is the enthalpy of CO2 using MACE medium at 500K?"
client = MultiServerMCPClient({
    "Chemistry Tools MCP": {
        "transport": "streamable_http",
        "url": "http://127.0.0.1:9001/mcp/",
    },
})


async def bootstrap():
    tools = await client.get_tools()
    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        structured_output=False,
        return_option="state",
        tools=tools,
    )
    result = await cg.run(prompt_single)
    print(result)
    """ Optional to print the entire state
    print("######## MESSAGE STATE #########")
    print(result)
    """


asyncio.run(bootstrap())
