import asyncio
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent_mcp import ChemGraph

#REMOTE_HOST = "YOUR_COMPUTE_NODE"
#REMOTE_ENV = "/path/to/venv"

REMOTE_HOST = "x4703c4s5b0n0"
REMOTE_ENV = "/lus/flare/projects/IQC/thang/ChemGraph/env/cg_frameworks_env"
current_dir = os.path.dirname(os.path.abspath(__file__))
MCP_SERVER = os.path.join(current_dir, "mcp_tools_stdio.py")

REMOTE_CMD = f"module load frameworks && source {REMOTE_ENV}/bin/activate && export http_proxy='proxy.alcf.anl.gov:3128' && export https_proxy='proxy.alcf.anl.gov:3128' && python -u {MCP_SERVER}"

prompt_single = "What is the enthalpy of CO2 using TBLite GFN2-xTB at 400K?"

client = MultiServerMCPClient({
    "Chemistry Tools MCP": {
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
    cg = ChemGraph(
        model_name="gpt-4o-mini",
        workflow_type="single_agent",
        structured_output=False,
        return_option="state",
        tools=tools,
    )
    result = await cg.run(prompt_single)

    """ Optional to print the entire state
    print("######## MESSAGE STATE #########")
    print(result)
    """


asyncio.run(bootstrap())
