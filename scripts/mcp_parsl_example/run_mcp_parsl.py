import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

prompt_single = "Run geometry optimization using MACE with cuda for the structures located in structures/. Save the output files in outputs/"

client = MultiServerMCPClient(
    {
        "Chemistry Tools MCP": {
            "transport": "streamable_http",
            "url": "http://127.0.0.1:9001/mcp/",
        },
    }
)


async def bootstrap():
    tools = await client.get_tools()
    # print(tools)
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
