#!/usr/bin/env python
"""Run ChemGraph's standard single agent with four MCP tool servers.

This example only handles MCP connection setup. ChemGraph owns the agent
prompt, LangGraph workflow, tool routing, and response generation.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

SERVER_MODULES = {
    "mofforge": "mofforge.mcp.server",
    "fairchem": "chemgraph.mcp.fairchem_mcp_hpc",
    "pacmof2": "chemgraph.mcp.pacmof2_mcp_hpc",
    "graspa": "chemgraph.mcp.graspa_mcp_hpc",
}

PYTHON_ENV_VARS = {
    name: f"{name.upper()}_PYTHON" for name in SERVER_MODULES
}

DEFAULT_QUERY = (
    "Use the available tools to list mofforge adsorbates and functional "
    "groups, then explain how FairChem, PACMOF2, and gRASPA would continue "
    "a MOF simulation workflow. Do not launch a simulation."
)

_ENV_NAMES = {
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONPATH",
    "XDG_CACHE_HOME",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}
_ENV_PREFIXES = (
    "CHEMGRAPH_",
    "GLOBUS_",
    "MOFFORGE_",
    "HF_",
    "CUDA_",
    "ZE_",
    "OMP_",
)


def _resolve_python(value: str | None, env_name: str) -> str:
    candidate = value or os.environ.get(env_name) or sys.executable
    resolved = shutil.which(candidate)
    if resolved is None:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            resolved = str(path)
    if resolved is None:
        raise ValueError(f"Python executable not found: {candidate!r}")
    return str(Path(resolved).resolve())


def _server_environment(backend: str, compute_system: str) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name in _ENV_NAMES or name.startswith(_ENV_PREFIXES)
    }
    env["CHEMGRAPH_EXECUTION_BACKEND"] = backend
    env["COMPUTE_SYSTEM"] = compute_system
    return env


def build_server_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Build stdio definitions for the four MCP servers."""
    env = _server_environment(args.backend, args.compute_system)
    configs = {}
    for name, module in SERVER_MODULES.items():
        python = _resolve_python(
            getattr(args, f"{name}_python"),
            PYTHON_ENV_VARS[name],
        )
        configs[name] = {
            "transport": "stdio",
            "command": python,
            "args": ["-u", "-m", module, "--transport", "stdio"],
            "env": dict(env),
        }
    return configs


def _prefix_tools(server_name: str) -> bool:
    """Keep mofforge_* names; namespace generic HPC job tools."""
    return server_name != "mofforge"


@contextlib.asynccontextmanager
async def persistent_tools(
    client: Any,
    server_names: list[str],
) -> AsyncIterator[list[Any]]:
    """Load tools while keeping their stdio server processes alive."""
    from langchain_mcp_adapters.tools import load_mcp_tools

    tools: list[Any] = []
    async with contextlib.AsyncExitStack() as stack:
        for server_name in server_names:
            session = await stack.enter_async_context(
                client.session(server_name)
            )
            tools.extend(
                await load_mcp_tools(
                    session,
                    server_name=server_name,
                    tool_name_prefix=_prefix_tools(server_name),
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
    parser.add_argument(
        "--backend",
        choices=["local", "parsl", "ensemble_launcher", "globus_compute"],
        default=os.environ.get("CHEMGRAPH_EXECUTION_BACKEND", "local"),
    )
    parser.add_argument(
        "--compute-system",
        default=os.environ.get("COMPUTE_SYSTEM", "local"),
    )
    for name in SERVER_MODULES:
        parser.add_argument(
            f"--{name}-python",
            default=None,
            help=f"Default: {PYTHON_ENV_VARS[name]} or current Python.",
        )
    parser.add_argument("--recursion-limit", type=int, default=50)
    parser.add_argument("--list-tools-only", action="store_true")
    return parser


def main() -> None:
    asyncio.run(amain(build_parser().parse_args()))


if __name__ == "__main__":
    main()
