#!/usr/bin/env python
"""Agent + MCP + Parsl demo running gRASPA over a directory of CIFs.

An LLM agent on the compute node drives a local ``graspa_mcp_hpc``
subprocess whose backend is ``parsl`` on Polaris/Aurora/Crux. The
agent uses ``run_graspa_ensemble`` to run GCMC on every ``.cif`` in the
supplied directory at a single (T, P) condition and reports a markdown
table.

Run inside ``qsub -I``. LLM API key required::

    export COMPUTE_SYSTEM=aurora
    export OPENAI_API_KEY=...
    python scripts/demo/demo_graspa_parsl_agent.py \\
        --model gpt-4o-mini --cif-dir /path/to/cifs --n-cycles 1000
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

from _demo_chemistry import agent_prompt_graspa, mcp_server_for

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from chemgraph.agent.llm_agent import ChemGraph


def _abort(msg: str) -> None:
    print(f"[ABORT] {msg}")
    sys.exit(2)


def _collect_cifs(cif_dir: str) -> list[str]:
    root = Path(cif_dir).expanduser().resolve()
    if not root.is_dir():
        _abort(f"--cif-dir {root} is not a directory.")
    cifs = sorted(str(p) for p in root.glob("*.cif"))
    if not cifs:
        _abort(f"No .cif files found in {root}.")
    return cifs


async def amain(
    model: str,
    system: str,
    query: str,
    verbose: int,
) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logging.getLogger("chemgraph").setLevel(logging.INFO if verbose == 1 else logging.DEBUG)

    server = mcp_server_for("graspa")
    server_label = f"{server['label']} (Parsl)"

    python = sys.executable
    env = {
        "CHEMGRAPH_EXECUTION_BACKEND": "parsl",
        "COMPUTE_SYSTEM": system,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "CONDA_PREFIX": os.environ.get("CONDA_PREFIX", ""),
        "CONDA_DEFAULT_ENV": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "CHEMGRAPH_WORKER_INIT": os.environ.get("CHEMGRAPH_WORKER_INIT", ""),
        "PBS_NODEFILE": os.environ.get("PBS_NODEFILE", ""),
        "PBS_O_WORKDIR": os.environ.get("PBS_O_WORKDIR", ""),
    }
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ARGO_API_KEY"):
        if os.environ.get(key):
            env[key] = os.environ[key]

    server_configs = {
        server_label: {
            "transport": "stdio",
            "command": python,
            "args": ["-u", "-m", server["module"]],
            "env": env,
        },
    }

    print(f"LLM model: {model}")
    print(f"MCP server: {server['module']}")
    print(f"Workload:  graspa")
    print(f"System:    {system}\n")
    print("Query:\n" + "-" * 60)
    print(query)
    print("-" * 60 + "\n")

    client = MultiServerMCPClient(server_configs)
    async with contextlib.AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session(server_label))
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
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--cif-dir", required=True,
                        help="Directory containing .cif files (node-reachable / shared FS).")
    parser.add_argument("--system", default=os.environ.get("COMPUTE_SYSTEM"))
    parser.add_argument("--adsorbate", default="H2O")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--pressure", type=float, default=101325.0)
    parser.add_argument("--query", default=None,
                        help="Override the auto-generated agent prompt.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if not os.environ.get("PBS_NODEFILE"):
        _abort("PBS_NODEFILE not set. Run inside `qsub -I`.")
    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora", "crux"):
        _abort(f"Unsupported --system: {system!r} (expected polaris|aurora|crux)")

    cif_paths = _collect_cifs(args.cif_dir)

    query = args.query or agent_prompt_graspa(
        cif_paths,
        adsorbate=args.adsorbate,
        temperature=args.temperature,
        pressure=args.pressure,
    )

    asyncio.run(amain(args.model, system, query, args.verbose))


if __name__ == "__main__":
    main()
