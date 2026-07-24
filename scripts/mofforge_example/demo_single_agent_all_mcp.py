#!/usr/bin/env python
"""Connect ChemGraph's standard single agent to four HTTP MCP servers.

Start the servers separately with ``start_mcp_servers.py``. This example only
handles MCP connection setup; ChemGraph owns the agent prompt, LangGraph
workflow, tool routing, and response generation.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
from collections.abc import AsyncIterator
from typing import Any

DEFAULT_SERVER_URLS = {
    "mofforge": "http://127.0.0.1:9010/mcp/",
    "fairchem": "http://127.0.0.1:9008/mcp/",
    "pacmof2": "http://127.0.0.1:9009/mcp/",
    "graspa": "http://127.0.0.1:9001/mcp/",
}

DEFAULT_QUERY = (
    "Use the available tools to list mofforge adsorbates and functional "
    "groups, then explain how FairChem, PACMOF2, and gRASPA would continue "
    "a MOF simulation workflow. Do not launch a simulation."
)

def build_server_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build streamable-HTTP definitions for already-running MCP servers."""
    configs: dict[str, dict[str, Any]] = {}
    for name in DEFAULT_SERVER_URLS:
        configs[name] = {
            "transport": "streamable_http",
            "url": getattr(args, f"{name}_url"),
        }
    return configs


def _prefix_tools(server_name: str) -> bool:
    """Keep mofforge_* names; namespace generic HPC job tools."""
    return server_name != "mofforge"


def _namespace_tools(server_name: str, tools: list[Any]) -> list[Any]:
    """Prefix LangChain-facing names without changing MCP call targets."""
    if _prefix_tools(server_name):
        for tool in tools:
            tool.name = f"{server_name}_{tool.name}"
    return tools


@contextlib.asynccontextmanager
async def persistent_tools(
    client: Any,
    server_names: list[str],
) -> AsyncIterator[list[Any]]:
    """Load tools while keeping the HTTP client sessions open."""
    from langchain_mcp_adapters.tools import load_mcp_tools

    tools: list[Any] = []
    async with contextlib.AsyncExitStack() as stack:
        for server_name in server_names:
            session = await stack.enter_async_context(
                client.session(server_name)
            )
            tools.extend(
                _namespace_tools(
                    server_name,
                    await load_mcp_tools(session),
                )
            )

        names = [tool.name for tool in tools]
        if len(names) != len(set(names)):
            raise RuntimeError("MCP servers advertised duplicate tool names")
        yield tools


async def run_chemgraph(
    tools: list[Any],
    *,
    model: str,
    query: str,
    recursion_limit: int,
) -> Any:
    """Run the existing ChemGraph single-agent workflow."""
    from chemgraph.agent.llm_agent import ChemGraph

    agent = ChemGraph(
        model_name=model,
        workflow_type="single_agent",
        structured_output=False,
        return_option="last_message",
        recursion_limit=recursion_limit,
        tools=tools,
        enable_memory=False,
    )
    return await agent.run(query)


async def amain(args: argparse.Namespace) -> None:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    configs = build_server_configs(args)
    client = MultiServerMCPClient(configs)

    async with persistent_tools(client, list(configs)) as tools:
        print(f"Loaded {len(tools)} MCP tools:")
        print(", ".join(sorted(tool.name for tool in tools)))
        if args.list_tools_only:
            return

        result = await run_chemgraph(
            tools,
            model=args.model,
            query=args.query,
            recursion_limit=args.recursion_limit,
        )
        print(f"\nChemGraph response:\n{result}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("CHEMGRAPH_MODEL", "argo:gpt-4o"),
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    for name, default_url in DEFAULT_SERVER_URLS.items():
        parser.add_argument(
            f"--{name}-url",
            default=default_url,
            help=f"Streamable-HTTP MCP endpoint (default: {default_url}).",
        )
    parser.add_argument("--recursion-limit", type=int, default=50)
    parser.add_argument("--list-tools-only", action="store_true")
    return parser


def main() -> None:
    asyncio.run(amain(build_parser().parse_args()))


if __name__ == "__main__":
    main()
