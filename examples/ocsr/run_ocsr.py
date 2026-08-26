"""Read structure diagrams and print their SMILES.

Two ways to use the OCSR tools, both runnable from this directory:

    python run_ocsr.py                  # call the tool directly, no agent, no LLM
    python run_ocsr.py --ensemble       # vote every specialist, with a confidence
    python run_ocsr.py --agent          # drive it through a ChemGraph agent

The direct path needs no API key. It calls ``image_to_smiles_core``, the same
function the tool wraps, so it exercises exactly what the agent would.

Every image here was drawn by RDKit from a known SMILES, so the script can check the
answer instead of only printing it. Agreement is by canonical form: a model may write
aspirin as ``CC(=O)OC1=CC=CC=C1C(=O)O`` or ``CC(=O)Oc1ccccc1C(=O)O`` and both are
right.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# The SMILES each image was drawn from. Not shown to any model.
EXPECTED = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "penicillin_g": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
}

IMAGES = pathlib.Path(__file__).parent / "images"


def _same_molecule(a: str | None, b: str) -> bool:
    """Compare by canonical SMILES, ignoring stereochemistry.

    Stereo-blind because these models disagree on wedge/dash reading far more often
    than on connectivity, and a stereo mismatch on penicillin should not read as
    "got the molecule wrong".
    """
    from rdkit import Chem

    if not a:
        return False
    mol_a, mol_b = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    if mol_a is None or mol_b is None:
        return False
    return Chem.MolToSmiles(mol_a, isomericSmiles=False) == Chem.MolToSmiles(
        mol_b, isomericSmiles=False
    )


def run_direct(model: str | None, ensemble: bool = False) -> int:
    """Call the tool on every image and report agreement with the known answer."""
    from chemgraph.tools.ocsr_backends import available_specialists
    from chemgraph.tools.ocsr_models import describe_models
    from chemgraph.tools.ocsr_tools import (image_to_smiles_core,
                                            measured_accuracies)

    installed = available_specialists()
    if ensemble and model:
        # The committee is specialists only, so it has no use for a named model.
        # Saying so beats running four votes the argument had no part in.
        print(f"--model {model} is ignored with --ensemble, which votes every "
              f"installed specialist.\n")
        model = None
    # The llm exemption does not apply to a committee: it never votes.
    if not installed and (ensemble or model != "llm"):
        print(describe_models(installed, measured_accuracies()))
        print("\nNo specialist is installed. Install one with:\n"
              "    pip install 'chemgraph[ocsr]'\n"
              "or read this image with the agent's own model:\n"
              "    python run_ocsr.py --agent --model llm")
        return 1

    print(describe_models(installed, measured_accuracies()), "\n")

    correct = 0
    for name, expected in EXPECTED.items():
        path = IMAGES / f"{name}.png"
        result = image_to_smiles_core(str(path), model=model, ensemble=ensemble)

        if not result["ok"]:
            print(f"  {name:14} FAILED  {result['error'][:90]}")
            continue

        hit = _same_molecule(result["smiles"], expected)
        correct += hit
        cold = " (cold start)" if result["cold_start"] else ""
        print(f"  {name:14} {'match' if hit else 'DIFFERS'}  "
              f"{result['smiles'][:58]}")
        print(f"  {'':14} {result['model_used']}, {result['latency_s']:.1f}s{cold}")
        if result["agreement"]:
            number = ("no number: " + (result["confidence_unavailable_reason"] or "")
                      if result["confidence"] is None
                      else f"{result['confidence']:.4f}")
            print(f"  {'':14} agreement {result['agreement']}, "
                  f"{number} ({result['confidence_label']})")
        if not hit:
            print(f"  {'':14} expected {expected[:58]}")

    print(f"\n  {correct}/{len(EXPECTED)} matched the structure each image was "
          f"drawn from.")
    return 0 if correct == len(EXPECTED) else 1


def run_agent(model_name: str, ocsr_model: str | None) -> int:
    """Ask a ChemGraph agent to read one image, so the tool call goes through an LLM."""
    from chemgraph.agent.llm_agent import ChemGraph

    agent = ChemGraph(model_name=model_name, workflow_type="ocsr")
    ask = f"What molecule is in {IMAGES / 'aspirin.png'}?"
    if ocsr_model:
        ask += f" Use the {ocsr_model} model."

    state = agent.run(ask)
    messages = state["messages"] if isinstance(state, dict) else state
    print(messages[-1].content if hasattr(messages[-1], "content") else messages[-1])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", action="store_true",
                        help="drive the tool through a ChemGraph agent (needs an LLM)")
    parser.add_argument("--model", default=None,
                        help="decimer, molnextr, molscribe, ocsrglyph, or llm. "
                             "Unset picks the default specialist.")
    parser.add_argument("--ensemble", action="store_true",
                        help="vote every installed specialist and report the "
                             "measured confidence instead of reading with one; "
                             "ignores --model, and needs a specialist installed")
    parser.add_argument("--llm", default="gpt-4o",
                        help="which LLM the agent runs on, with --agent")
    args = parser.parse_args()

    if args.agent:
        return run_agent(args.llm, args.model)
    return run_direct(args.model, ensemble=args.ensemble)


if __name__ == "__main__":
    sys.exit(main())
