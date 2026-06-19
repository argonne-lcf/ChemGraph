#!/usr/bin/env python
"""Agent + MCP demo on LocalBackend: LLM screens 5 molecules locally.

Spawns ``chemgraph.mcp.mace_mcp_hpc`` as a local subprocess wired to
the LocalBackend, then asks the ChemGraph LLM agent to compute
thermochemistry on water / methane / ammonia / CO2 / ethanol via the
MCP ``run_mace_single`` tool and report a markdown table.

Prereq: an LLM API key for the chosen model (e.g. ``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, Argo gateway tokens via ``inference_auth_token.py``,
etc.) and ``langchain-mcp-adapters`` installed (already a dep).

Run::

    export OPENAI_API_KEY=...
    python scripts/demo/demo_local_agent.py --model gpt-4o-mini
    python scripts/demo/demo_local_agent.py --model argo:gpt-4o
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

from _demo_chemistry import agent_prompt

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from chemgraph.agent.llm_agent import ChemGraph


async def amain(model: str, device: str, query: str, verbose: int) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logging.getLogger("chemgraph").setLevel(logging.INFO if verbose == 1 else logging.DEBUG)

    # Make sure the spawned MCP subprocess uses LocalBackend.
    os.environ["CHEMGRAPH_EXECUTION_BACKEND"] = "local"

    python = sys.executable
    server_configs = {
        "ChemGraph MACE": {
            "transport": "stdio",
            "command": python,
            "args": ["-u", "-m", "chemgraph.mcp.mace_mcp_hpc"],
            "env": {
                "CHEMGRAPH_EXECUTION_BACKEND": "local",
                # Forward the user's PATH/HOME so the subprocess can resolve
                # the venv's chemgraph + mace_torch installs.
                "PATH": os.environ.get("PATH", ""),
                "HOME": os.environ.get("HOME", ""),
                "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
            },
        },
    }

    print(f"LLM model:   {model}")
    print(f"MCP server:  mace_mcp_hpc (stdio subprocess, CHEMGRAPH_EXECUTION_BACKEND=local)")
    print(f"Device:      {device}\n")
    print("Query:\n" + "-" * 60)
    print(query)
    print("-" * 60 + "\n")

    client = MultiServerMCPClient(server_configs)

    async with contextlib.AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session("ChemGraph MACE"))
        tools = await load_mcp_tools(session)
        tool_names = [t.name for t in tools]
        print(f"Loaded {len(tools)} MCP tools: {tool_names}\n")

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
        help="LLM model name (default: argo:gpt-4o). Try argo:gpt-4o, claude-sonnet-4-6, gpt-4o.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="MACE device passed to the agent prompt (default: cpu)",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Override the natural-language query (default: 5-molecule thermo screen)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v INFO, -vv DEBUG).",
    )
    args = parser.parse_args()

    query = args.query or agent_prompt(device=args.device)
    asyncio.run(amain(args.model, args.device, query, args.verbose))


if __name__ == "__main__":
    main()
