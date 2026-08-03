"""Run molecular docking through the ChemGraph agent.

Docks a small-molecule candidate into a receptor with AutoDock Vina, driven by the
ChemGraph single-agent graph bound to the ``run_docking`` tool and a docking-specific
prompt. The candidate can be a SMILES, a molecule name, or a PubChem CID; the receptor
here is the prepared vancomycin PDBQT that ships alongside this example.

Prerequisites:
  - An LLM provider key (e.g. OPENAI_API_KEY)
  - The docking extra + AutoDock Vina:
      pip install -e ".[docking]"
      conda install -c conda-forge vina

Usage:
  export OPENAI_API_KEY="your_key"
  python run_chemgraph.py
"""

import asyncio
import os

from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.prompt.docking_prompt import docking_prompt
from chemgraph.tools.docking_tools import run_docking

MODEL_NAME = "gpt-4o-mini"

# A prepared vancomycin receptor (PDB 1FVM, chain A) ships alongside this example.
RECEPTOR = os.path.join(os.path.dirname(__file__), "vancomycin_receptor.pdbqt")

# Try changing the candidate (SMILES / name / PubChem CID) or n_poses.
PROMPT = (
    f"Dock aspirin into the receptor at '{RECEPTOR}' using 10 poses, "
    "and report the best binding affinity."
)


async def main():
    cg = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type="single_agent",
        system_prompt=docking_prompt,
        tools=[run_docking],
    )
    print(f"Model: {MODEL_NAME}")
    print(f"Prompt: {PROMPT}\n")
    result = await cg.run(PROMPT)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
