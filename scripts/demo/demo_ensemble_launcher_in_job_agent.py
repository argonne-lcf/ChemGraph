#!/usr/bin/env python
"""Agent + MCP + EnsembleLauncher demo on an HPC compute node.

LLM agent on the compute node drives a local ``mace_mcp_hpc``
subprocess whose backend is ``ensemble_launcher``. Same 5-molecule
thermo screen as the direct demo, but driven natural-language.

Run inside ``qsub -I`` on Polaris/Aurora. LLM API key required.

Run::

    export COMPUTE_SYSTEM=polaris
    export OPENAI_API_KEY=...
    python scripts/demo/demo_ensemble_launcher_in_job_agent.py --model gpt-4o-mini
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


def _abort(msg: str) -> None:
    print(f"[ABORT] {msg}")
    sys.exit(2)


async def amain(model: str, system: str, device: str, query: str, verbose: int,
                *, ppn: int = 1, ngpus_per_process: int = 0) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logging.getLogger("chemgraph").setLevel(logging.INFO if verbose == 1 else logging.DEBUG)

    python = sys.executable
    env = {
        "CHEMGRAPH_EXECUTION_BACKEND": "ensemble_launcher",
        "COMPUTE_SYSTEM": system,
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
        "PBS_NODEFILE": os.environ.get("PBS_NODEFILE", ""),
        "PBS_O_WORKDIR": os.environ.get("PBS_O_WORKDIR", ""),
    }
    server_configs = {
        "ChemGraph MACE (EnsembleLauncher)": {
            "transport": "stdio",
            "command": python,
            "args": ["-u", "-m", "chemgraph.mcp.mace_mcp_hpc",
                    "--ppn", str(ppn),
                    "--ngpus-per-process", str(ngpus_per_process)],
            "env": env,
        },
    }

    print(f"LLM model: {model}")
    print(f"System:    {system}")
    print(f"Device:    {device}\n")
    print("Query:\n" + "-" * 60)
    print(query)
    print("-" * 60 + "\n")

    client = MultiServerMCPClient(server_configs)
    async with contextlib.AsyncExitStack() as stack:
        session = await stack.enter_async_context(
            client.session("ChemGraph MACE (EnsembleLauncher)")
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
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--system", default=os.environ.get("COMPUTE_SYSTEM"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--ppn", type=int, default=1,
                        help="Processes per node for MCP backend tasks")
    parser.add_argument("--ngpus-per-process", type=int, default=0,
                        help="GPUs per process for MCP backend tasks")
    parser.add_argument("--query", default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    if not os.environ.get("PBS_NODEFILE"):
        _abort("PBS_NODEFILE not set. Run inside `qsub -I`.")
    if not args.system:
        _abort("COMPUTE_SYSTEM env var not set and --system not given.")
    system = args.system.lower().strip()
    if system not in ("polaris", "aurora"):
        _abort(f"Unsupported --system: {system!r}")
    device = args.device or ("xpu" if system == "aurora" else "cuda")
    query = args.query or agent_prompt(device=device)
    asyncio.run(amain(args.model, system, device, query, args.verbose,
                      ppn=args.ppn, ngpus_per_process=args.ngpus_per_process))


if __name__ == "__main__":
    main()
