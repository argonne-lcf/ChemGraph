#!/usr/bin/env python
"""Expose mofforge, FairChem, PACMOF2, and gRASPA tools to one agent.

The four MCP servers run as persistent stdio subprocesses.  Keeping their
sessions open for the full agent run is important: the ChemGraph HPC servers
store local futures and job-tracker state in their server processes.

Examples
--------
List the available tools without calling an LLM::

    python scripts/mofforge_example/demo_single_agent_all_mcp.py \
        --list-tools-only

Run the default lightweight agent query::

    python scripts/mofforge_example/demo_single_agent_all_mcp.py \
        --model argo:gpt-4o

Use separate environments for optional worker dependencies::

    FAIRCHEM_PYTHON=.cg_uma_env/bin/python \
    PACMOF2_PYTHON=/path/to/pacmof2-env/bin/python \
    GRASPA_PYTHON=/path/to/graspa-env/bin/python \
    python scripts/mofforge_example/demo_single_agent_all_mcp.py \
        --model argo:gpt-4o
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
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
    "mofforge": "MOFFORGE_PYTHON",
    "fairchem": "FAIRCHEM_PYTHON",
    "pacmof2": "PACMOF2_PYTHON",
    "graspa": "GRASPA_PYTHON",
}

REQUIRED_TOOLS = {
    "mofforge": {
        "mofforge_build",
        "mofforge_validate",
    },
    "fairchem": {
        "fairchem_run_fairchem_single",
        "fairchem_run_fairchem_ensemble",
    },
    "pacmof2": {
        "pacmof2_run_pacmof2_ensemble",
    },
    "graspa": {
        "graspa_run_graspa_ensemble",
    },
}

_FORWARDED_ENV_NAMES = {
    "PATH",
    "HOME",
    "USER",
    "TMPDIR",
    "LANG",
    "LC_ALL",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "PYTHONPATH",
    "XDG_CACHE_HOME",
    "COMPUTE_SYSTEM",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
}

_FORWARDED_ENV_PREFIXES = (
    "CHEMGRAPH_",
    "GLOBUS_",
    "MOFFORGE_",
    "HF_",
    "CUDA_",
    "ZE_",
    "OMP_",
)

DEFAULT_QUERY = (
    "Use the mofforge tools to list the built-in adsorbates and functional "
    "groups. Then, without launching a simulation, explain which exposed "
    "FairChem, PACMOF2, and gRASPA tools would take a selected MOF from "
    "structural relaxation through partial-charge assignment to adsorption "
    "simulation. Base the answer on actual tool outputs and advertised tools."
)

MOF_AGENT_PROMPT = """\
You are a computational materials agent for metal-organic framework workflows.

All claims about generated structures or simulation results must come from tool
outputs. Never invent paths, structures, batch identifiers, or numerical
results.

The available tools are grouped by engine:
- mofforge_* tools search, build, modify, functionalize, render, and validate
  MOF structures.
- fairchem_* tools relax structures or calculate energies with FairChem/UMA.
- pacmof2_* tools assign partial atomic charges to CIF files.
- graspa_* tools run adsorption simulations on charged CIF files.

Preserve absolute CIF and result paths returned by tools when passing outputs
between engines. For an end-to-end calculation, use the order requested by the
user; the usual sequence is mofforge structure preparation, FairChem
relaxation, PACMOF2 charge assignment, and gRASPA adsorption.

ChemGraph backend tools can return status='submitted' with a batch_id. Poll and
collect that job only with tools carrying the same engine prefix. For example,
use fairchem_check_job_status and fairchem_get_job_results for a FairChem batch,
never the PACMOF2 or gRASPA job tools. If work remains pending, report the
engine and batch_id instead of fabricating a result.
"""


def _resolve_python(value: str | None, env_name: str) -> str:
    """Resolve and validate a Python interpreter for an MCP subprocess."""
    candidate = value or os.environ.get(env_name) or sys.executable
    resolved = shutil.which(candidate)
    if resolved is None:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            resolved = str(path.resolve())
    if resolved is None:
        raise ValueError(
            f"Could not find executable {candidate!r}. Set {env_name} or pass "
            f"the corresponding --*-python option."
        )
    return str(Path(resolved).resolve())


def _forwarded_environment(backend: str, compute_system: str) -> dict[str, str]:
    """Build the restricted environment inherited by MCP subprocesses."""
    child_env = {
        name: value
        for name, value in os.environ.items()
        if name in _FORWARDED_ENV_NAMES
        or name.startswith(_FORWARDED_ENV_PREFIXES)
    }
    child_env["CHEMGRAPH_EXECUTION_BACKEND"] = backend
    child_env["COMPUTE_SYSTEM"] = compute_system
    return child_env


def _tool_prefix_enabled(server_name: str) -> bool:
    """Prefix tools whose servers advertise generic job-management names."""
    return server_name != "mofforge"


def build_server_configs(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """Create the four stdio connection definitions."""
    env = _forwarded_environment(args.backend, args.compute_system)
    configs: dict[str, dict[str, Any]] = {}

    for server_name, module in SERVER_MODULES.items():
        python_value = getattr(args, f"{server_name}_python")
        python = _resolve_python(python_value, PYTHON_ENV_VARS[server_name])
        server_args = ["-u", "-m", module]
        if server_name == "fairchem":
            server_args.extend(
                [
                    "--ppn",
                    str(args.fairchem_ppn),
                    "--ngpus-per-process",
                    str(args.fairchem_ngpus_per_process),
                ]
            )
        server_args.extend(["--transport", "stdio"])
        configs[server_name] = {
            "transport": "stdio",
            "command": python,
            "args": server_args,
            "env": dict(env),
        }

    return configs


def validate_tool_inventory(grouped_tools: dict[str, list[Any]]) -> None:
    """Reject ambiguous or incomplete MCP tool inventories."""
    owners: dict[str, str] = {}
    duplicates: list[str] = []

    for server_name, tools in grouped_tools.items():
        names = {tool.name for tool in tools}
        missing = sorted(REQUIRED_TOOLS[server_name] - names)
        if missing:
            raise RuntimeError(
                f"MCP server {server_name!r} did not advertise required tools: "
                f"{missing}. Advertised: {sorted(names)}"
            )
        for name in names:
            if name in owners:
                duplicates.append(
                    f"{name!r} from {owners[name]!r} and {server_name!r}"
                )
            owners[name] = server_name

    if duplicates:
        raise RuntimeError(
            "Duplicate MCP tool names remain after namespacing: "
            + "; ".join(sorted(duplicates))
        )


@contextlib.asynccontextmanager
async def persistent_mcp_tools(
    client: Any,
    server_names: list[str],
) -> AsyncIterator[tuple[list[Any], dict[str, list[Any]]]]:
    """Load tools while keeping every server session alive."""
    from langchain_mcp_adapters.tools import load_mcp_tools

    all_tools: list[Any] = []
    grouped_tools: dict[str, list[Any]] = {}

    async with contextlib.AsyncExitStack() as stack:
        for server_name in server_names:
            try:
                session = await stack.enter_async_context(
                    client.session(server_name)
                )
                tools = await load_mcp_tools(
                    session,
                    server_name=server_name,
                    tool_name_prefix=_tool_prefix_enabled(server_name),
                )
            except Exception as exc:
                module = SERVER_MODULES[server_name]
                raise RuntimeError(
                    f"Failed to start/load MCP server {server_name!r} "
                    f"({module}): {exc}"
                ) from exc
            grouped_tools[server_name] = tools
            all_tools.extend(tools)

        validate_tool_inventory(grouped_tools)
        yield all_tools, grouped_tools


def _print_tool_inventory(grouped_tools: dict[str, list[Any]]) -> None:
    total = sum(len(tools) for tools in grouped_tools.values())
    print(f"\nLoaded {total} MCP tools across {len(grouped_tools)} servers:")
    for server_name, tools in grouped_tools.items():
        print(f"\n[{server_name}] ({len(tools)})")
        for name in sorted(tool.name for tool in tools):
            print(f"  - {name}")


def _final_response(result: Any) -> str:
    """Extract the last non-tool response from a ChemGraph state."""
    if not isinstance(result, dict):
        return str(result)

    for message in reversed(result.get("messages", [])):
        if isinstance(message, dict):
            content = message.get("content")
            tool_calls = message.get("tool_calls")
        else:
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None)
        if content and not tool_calls:
            return str(content)
    return str(result)


def _configure_logging(verbosity: int) -> None:
    if verbosity <= 0:
        return
    level = logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )


async def amain(args: argparse.Namespace) -> None:
    """Connect the four MCP servers and optionally run the single agent."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    _configure_logging(args.verbose)
    server_configs = build_server_configs(args)
    client = MultiServerMCPClient(server_configs)

    print(f"Execution backend: {args.backend}")
    print(f"Compute system:    {args.compute_system}")
    print("Starting persistent MCP stdio sessions...")

    async with persistent_mcp_tools(
        client, list(server_configs)
    ) as (tools, grouped_tools):
        _print_tool_inventory(grouped_tools)
        if args.list_tools_only:
            print("\nOK: all required MCP tool surfaces are available.")
            return

        from chemgraph.agent.llm_agent import ChemGraph

        print(f"\nLLM model: {args.model}")
        print(f"Query: {args.query}\n")
        agent = ChemGraph(
            model_name=args.model,
            workflow_type="single_agent",
            system_prompt=MOF_AGENT_PROMPT,
            structured_output=False,
            return_option="state",
            recursion_limit=args.recursion_limit,
            tools=tools,
            enable_memory=False,
            log_dir=args.log_dir,
        )
        result = await agent.run(args.query)
        print(f"\nAgent response:\n{_final_response(result)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("CHEMGRAPH_MODEL", "argo:gpt-4o"),
        help="ChemGraph LLM model (default: CHEMGRAPH_MODEL or argo:gpt-4o).",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Natural-language task for the single agent.",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "parsl", "ensemble_launcher", "globus_compute"],
        default=os.environ.get("CHEMGRAPH_EXECUTION_BACKEND", "local"),
        help="Execution backend used by ChemGraph MCP servers.",
    )
    parser.add_argument(
        "--compute-system",
        default=os.environ.get("COMPUTE_SYSTEM", "local"),
        help="Compute-system profile used by the selected backend.",
    )
    for server_name in SERVER_MODULES:
        option = f"--{server_name}-python"
        parser.add_argument(
            option,
            default=None,
            help=(
                f"Python for the {server_name} MCP server "
                f"(default: {PYTHON_ENV_VARS[server_name]} or current Python)."
            ),
        )
    parser.add_argument(
        "--fairchem-ppn",
        type=int,
        default=1,
        help="Processes per node requested by FairChem tools.",
    )
    parser.add_argument(
        "--fairchem-ngpus-per-process",
        type=int,
        default=0,
        help="GPUs per FairChem worker process.",
    )
    parser.add_argument(
        "--recursion-limit",
        type=int,
        default=50,
        help="Maximum LangGraph recursion steps.",
    )
    parser.add_argument(
        "--log-dir",
        default=os.environ.get("CHEMGRAPH_LOG_DIR"),
        help="Directory for ChemGraph run artifacts.",
    )
    parser.add_argument(
        "--list-tools-only",
        action="store_true",
        help="Load and print all tools without creating an LLM agent.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        asyncio.run(amain(args))
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
