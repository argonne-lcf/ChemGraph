"""Read a molecule's structure diagram and return its SMILES.

Two ways to use the OCSR tool, both shown here:

1. Directly, as a plain function. No agent, no LLM, no credentials. This is the one
   to reach for when you just want a SMILES out of a picture.
2. Through a ChemGraph agent, which is what makes it compose with the rest of the
   toolbox: read the structure, then build a 3D geometry, then optimize it, in one
   conversation.

Run:
    python run_chemgraph.py
    python run_chemgraph.py --agent          # needs an LLM that can call tools
"""

from __future__ import annotations

import argparse
import asyncio
import os

IMAGE = os.path.join(os.path.dirname(__file__), "aspirin.png")


def show(result: dict) -> None:
    """Print the fields worth looking at, in the order a human reads them."""
    if not result["ok"]:
        print(f"  failed: {result['error']}")
        return
    conf = result["confidence"]
    conf_str = f"{conf:.3f}" if conf is not None else "n/a"
    print(f"  SMILES     {result['smiles']}")
    print(f"  formula    {result['formula']}")
    print(f"  confidence {conf_str}  ({result['confidence_label']}, "
          f"basis={result['basis']}, agreement={result['agreement']})")
    print(f"  backend    {result['backend_used']} / {result['model_used']}"
          f"   {result['latency_s']:.1f}s"
          f"{' (cold start, models were loading)' if result['cold_start'] else ''}")
    if result["votes"]:
        print("  votes")
        for smiles, voters in result["votes"].items():
            print(f"    {len(voters)}x {smiles}   {', '.join(voters)}")
    if result["abstained"]:
        print(f"  abstained  {', '.join(result['abstained'])}")
    if result["warning"]:
        print(f"  WARNING    {result['warning']}")


def direct() -> None:
    from chemgraph.tools.ocsr_models import describe_backends
    from chemgraph.tools.ocsr_tools import image_to_smiles_core as read_structure

    print(describe_backends())
    print()

    print("=== one specialist model (the default) ===")
    print("(no confidence: one model cannot say how likely it is to be right here)")
    show(read_structure(IMAGE))

    print("\n=== the same model, with its overall benchmark accuracy ===")
    print("(a property of the model, not of this image; opt in when that is useful)")
    show(read_structure(IMAGE, report_solo_accuracy=True))

    print("\n=== all four, voted, with a calibrated confidence ===")
    print("(slower, and the only backend that scores THIS image)")
    show(read_structure(IMAGE, backend="ensemble"))

    print("\n=== two of the four, for a committee you calibrated yourself ===")
    show(read_structure(IMAGE, backend="ensemble",
                        models_wanted=["decimer", "molnextr"]))

    print("\n=== a vision LLM instead, if you have a token ===")
    show(read_structure(IMAGE, backend="alcf"))


async def agent() -> None:
    """The same tool through the ocsr workflow, which binds it for you."""
    from chemgraph.agent.llm_agent import ChemGraph

    cg = ChemGraph(
        model_name=os.environ.get("CHEMGRAPH_MODEL", "gpt-4o-mini"),
        workflow_type="ocsr",
        enable_memory=False,
    )
    out = await cg.run(f"What molecule is in {IMAGE}? Report how confident you are.")
    print(getattr(out, "content", out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agent", action="store_true",
                   help="drive the tool through a ChemGraph agent (needs an LLM)")
    args = p.parse_args()
    if args.agent:
        asyncio.run(agent())
    else:
        direct()
