#!/usr/bin/env python
"""Agent + MCP + Globus Compute demo: 5-molecule thermo screen on remote HPC.

LLM agent on the laptop, MCP server (``mace_mcp_hpc``) as a local
subprocess, work dispatched to a Globus Compute endpoint on Polaris /
Aurora. Mirrors ``scripts/globus_compute_example/run_agent_mcp_remote.py``
but with a structured 5-molecule chemistry workload instead of a free
prompt.

Prereqs::

    export GLOBUS_COMPUTE_ENDPOINT_ID="<uuid>"
    export OPENAI_API_KEY=...                       # or other model creds

Run::

    python scripts/demo/demo_globus_compute_agent.py --model gpt-4o-mini
    python scripts/demo/demo_globus_compute_agent.py --device xpu --model argo:gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _demo_chemistry import mcp_server_for, prompt_for

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from chemgraph.agent.llm_agent import ChemGraph


async def amain(model: str, device: str, query: str, workload: str, verbose: int) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logging.getLogger("chemgraph").setLevel(logging.INFO if verbose == 1 else logging.DEBUG)

    endpoint = os.environ["GLOBUS_COMPUTE_ENDPOINT_ID"]
    env = os.environ.copy()
    env.update({
        "CHEMGRAPH_EXECUTION_BACKEND": "globus_compute",
    })

    server = mcp_server_for(workload)
    server_label = f"{server['label']} (Globus Compute)"

    python = sys.executable
    server_configs = {
        server_label: {
            "transport": "stdio",
            "command": python,
            "args": ["-u", "-m", server["module"]],
            "env": env
        },
    }

    print(f"LLM model:    {model}")
    print(f"MCP server:   {server['module']}")
    print(f"Workload:     {workload}")
    print(f"GC endpoint:  {endpoint[:8]}... ({os.environ.get('COMPUTE_SYSTEM', '?')})")
    print(f"Device:       {device}\n")
    print("Query:\n" + "-" * 60)
    print(query)
    print("-" * 60 + "\n")

    client = MultiServerMCPClient(server_configs)

    async with contextlib.AsyncExitStack() as stack:
        session = await stack.enter_async_context(
            client.session(server_label)
        )
        tools = await load_mcp_tools(session)
        print(f"Loaded {len(tools)} MCP tools: {[t.name for t in tools]}\n")

        cg = ChemGraph(
            model_name=model,
            workflow_type="single_agent",
            structured_output=False,
            return_option="state",
            tools=tools,
        )

        print("Running agent...\n" + "=" * 60)
        result = await cg.run(query)
        print("=" * 60)

        if isinstance(result, dict) and "messages" in result:
            for msg in reversed(result["messages"]):
                content = getattr(msg, "content", None)
                if not content and isinstance(msg, dict):
                    content = msg.get("content", "")
                if content and not getattr(msg, "tool_calls", None):
                    print(f"\nAgent response:\n{content}")
                    break
        else:
            print(f"\nResult:\n{result}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="argo:gpt-4o",
        help="LLM model name (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("CG_DEMO_DEVICE", "cuda"),
        help="MACE/ASE device on the remote endpoint (default: cuda; use xpu on Aurora)",
    )
    parser.add_argument(
        "--workload",
        choices=["thermo", "ase", "graspa"],
        default="thermo",
        help="thermo = MACE; ase = general ASE; graspa = GCMC (needs --graspa-cifs).",
    )
    parser.add_argument(
        "--calculator",
        default="mace_mp",
        help="ASE calculator for --workload ase (e.g. mace_mp, emt, tblite).",
    )
    parser.add_argument(
        "--graspa-cifs",
        nargs="+",
        default=None,
        help="Remote-reachable CIF paths for --workload graspa.",
    )
    parser.add_argument("--query", default=None, help="Override the default query")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if not os.environ.get("GLOBUS_COMPUTE_ENDPOINT_ID"):
        print("ERROR: export GLOBUS_COMPUTE_ENDPOINT_ID=<uuid> first.")
        sys.exit(2)

    if args.query:
        query = args.query
    elif args.workload == "graspa":
        from _demo_chemistry import agent_prompt_graspa

        if not args.graspa_cifs:
            print("ERROR: --workload graspa requires --graspa-cifs <CIF> [<CIF> ...].")
            sys.exit(2)
        query = agent_prompt_graspa(args.graspa_cifs)
    else:
        query = prompt_for(args.workload, device=args.device, calculator=args.calculator)

    asyncio.run(amain(args.model, args.device, query, args.workload, args.verbose))


if __name__ == "__main__":
    main()
