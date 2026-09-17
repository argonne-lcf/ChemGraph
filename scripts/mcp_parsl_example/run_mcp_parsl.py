import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

prompt_single = (
    "Use run_graspa_ensemble for H2O adsorption on the shared CIF directory "
    "structures/ at 298.15 K and 1000 Pa. Set output_directory='water-screening'. "
    "If a batch is submitted, poll check_job_status and retrieve get_job_results. "
    "Report uptake in mol/kg, failures, and actual returned artifact paths."
)

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
