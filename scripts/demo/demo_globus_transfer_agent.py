#!/usr/bin/env python
"""Agent + MCP + Globus Transfer + Globus Compute demo.

LLM agent on the laptop drives a local ``mace_mcp_hpc`` subprocess.
With both Compute and Transfer env vars set, the MCP server
auto-registers the transfer tools (``mace_mcp_hpc.py:310-313``). The
agent is told to (a) stage the demo's structures to the remote
collection via ``transfer_files``, then (b) call ``run_mace_ensemble``
with ``remote_structure_directory`` so MACE runs on the pre-staged
files. Finally it reports a Gibbs-energy table.

Prereqs::

    export GLOBUS_COMPUTE_ENDPOINT_ID=...
    export GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID=...
    export GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID=...
    export GLOBUS_TRANSFER_DESTINATION_BASE_PATH=/eagle/projects/MyProj/staging
    export OPENAI_API_KEY=...      # or any supported model

Run::

    python scripts/demo/demo_globus_transfer_agent.py --model gpt-4o-mini
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

from _demo_chemistry import MOLECULE_NAMES, structures_dir

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from chemgraph.agent.llm_agent import ChemGraph


_TRANSFER_AGENT_PROMPT_TMPL = """\
The following five molecule structure files live on the local filesystem:
{listing}

Workflow:
1. Call `transfer_files` with `source_paths` set to that list of absolute
   paths (you may pass them as one batch) to stage them on the remote
   HPC endpoint. Use `wait=true` so the call blocks until SUCCEEDED.
2. From the transfer result, take the `remote_directory` value.
3. Call `run_mace_ensemble` with:
     - remote_structure_directory = <remote_directory from step 2>
     - driver = "thermo"
     - model = "medium-mpa-0"
     - device = "{device}"
     - temperature = 298.15
     - pressure = 101325
   This dispatches one MACE thermo job per file via Globus Compute.
4. If `run_mace_ensemble` returns a `batch_id`, poll `check_job_status`
   until completed, then call `get_job_results` to retrieve the per-file
   energies and thermochemistry.
5. Report a markdown table with columns: molecule | electronic energy (eV) |
   Gibbs free energy (eV). Add a one-line observation about which
   molecule has the lowest Gibbs free energy.
"""


def _agent_prompt(device: str) -> str:
    paths = [str(structures_dir() / f"{n}.xyz") for n in MOLECULE_NAMES]
    listing = "\n".join(f"  - {p}" for p in paths)
    return _TRANSFER_AGENT_PROMPT_TMPL.format(listing=listing, device=device)


async def amain(model: str, device: str, query: str, verbose: int) -> None:
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
        logging.getLogger("chemgraph").setLevel(logging.INFO if verbose == 1 else logging.DEBUG)

    python = sys.executable
    forwarded = {
        "CHEMGRAPH_EXECUTION_BACKEND": "globus_compute",
        "GLOBUS_COMPUTE_ENDPOINT_ID": os.environ["GLOBUS_COMPUTE_ENDPOINT_ID"],
        "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID": os.environ["GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID"],
        "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID": os.environ["GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID"],
        "GLOBUS_TRANSFER_DESTINATION_BASE_PATH": os.environ["GLOBUS_TRANSFER_DESTINATION_BASE_PATH"],
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "VIRTUAL_ENV": os.environ.get("VIRTUAL_ENV", ""),
    }
    server_configs = {
        "ChemGraph MACE+Transfer": {
            "transport": "stdio",
            "command": python,
            "args": ["-u", "-m", "chemgraph.mcp.mace_mcp_hpc"],
            "env": forwarded,
        },
    }

    print(f"LLM model: {model}")
    print(f"Device:    {device}\n")
    print("Query:\n" + "-" * 60)
    print(query)
    print("-" * 60 + "\n")

    client = MultiServerMCPClient(server_configs)
    async with contextlib.AsyncExitStack() as stack:
        session = await stack.enter_async_context(client.session("ChemGraph MACE+Transfer"))
        tools = await load_mcp_tools(session)
        names = [t.name for t in tools]
        print(f"Loaded {len(tools)} MCP tools: {names}\n")
        if "transfer_files" not in names:
            print(
                "WARNING: transfer_files not registered. Did you export the "
                "GLOBUS_TRANSFER_* env vars? mace_mcp_hpc only registers the "
                "transfer tools when a transfer manager is configured."
            )

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
    parser.add_argument("--device", default=os.environ.get("CG_DEMO_DEVICE", "cuda"))
    parser.add_argument("--query", default=None)
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args()

    required = (
        "GLOBUS_COMPUTE_ENDPOINT_ID",
        "GLOBUS_TRANSFER_SOURCE_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_ENDPOINT_ID",
        "GLOBUS_TRANSFER_DESTINATION_BASE_PATH",
    )
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print(f"ERROR: missing env vars: {', '.join(missing)}")
        sys.exit(2)

    query = args.query or _agent_prompt(args.device)
    asyncio.run(amain(args.model, args.device, query, args.verbose))


if __name__ == "__main__":
    main()
