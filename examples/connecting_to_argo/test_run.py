"""Smoke test: ChemGraph → argo-shim → Argo, from an ALCF compute node.

Prereqs (see README.md):
  1. Login-node argo-shim tunnel is up on some UAN:
       argo-shim --no-auth --no-update-settings --port 18085 --tunnel
  2. Compute-node argo-shim points at that UAN:
       argo-shim --no-auth --no-update-settings \\
         --port 18085 --tunnel-host <UAN> --tunnel-port 18084
     with no_proxy=<UAN>,127.0.0.1,localhost,.alcf.anl.gov,*.alcf.anl.gov

Run:
  python examples/connecting_to_argo/test_run.py

Override defaults via env vars if needed:
  ARGO_USER   -- your CELS login (e.g. jane.doe)
  ARGO_MODEL  -- argo model name (default: argo:gpt-4.1-mini)
  ARGO_BASE   -- shim URL (default: http://127.0.0.1:18085/argoapi/v1)
  ARGO_PROMPT -- override the prompt

Expected: LLM answers the query, possibly by calling `run_ase` or
`molecule_name_to_smiles` first.
"""
from __future__ import annotations

import asyncio
import os
import sys

# argo-shim rejects the stripped lowercase-hyphenated form (e.g.
# "gpt-4.1-mini") with 400 Invalid model. Tell ChemGraph's normalizer
# to send the wire form (e.g. "gpt41mini"), which the shim also
# accepts, so we don't need a per-model display-case map. Set BEFORE
# importing chemgraph so the normalizer picks it up.
os.environ.setdefault("CHEMGRAPH_ARGO_MODEL_FORMAT", "wire")


def main() -> int:
    from chemgraph.agent.llm_agent import ChemGraph

    argo_user = os.environ.get("ARGO_USER")
    if not argo_user:
        print(
            "ERROR: set ARGO_USER to your CELS login "
            "(e.g. `export ARGO_USER=jane.doe`).",
            file=sys.stderr,
        )
        return 1

    model = os.environ.get("ARGO_MODEL", "argo:gpt-4.1-mini")
    base_url = os.environ.get(
        "ARGO_BASE", "http://127.0.0.1:18085/argoapi/v1",
    )
    prompt = os.environ.get(
        "ARGO_PROMPT",
        "Run geometry optimization for water using TBLite",
    )

    if not model.startswith("argo:"):
        print(
            "WARNING: model_name should start with 'argo:' for the shim "
            "path. Got:",
            model,
            file=sys.stderr,
        )

    print(f"Model:    {model}")
    print(f"Base URL: {base_url}")
    print(f"Argo user: {argo_user}")
    print(f"Prompt:   {prompt}")
    print()

    cg = ChemGraph(
        model_name=model,
        workflow_type="single_agent",
        base_url=base_url,
        api_key="dummy",  # argo-shim doesn't check
        argo_user=argo_user,
    )
    result = asyncio.run(cg.run(prompt))
    print()
    print("=" * 60)
    print("ChemGraph response:")
    print("=" * 60)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
