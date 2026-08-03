"""LangChain ``@tool`` wrapper for molecular docking.

Delegates to the pure-Python implementation in
:mod:`chemgraph.tools.docking_core`.
"""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.docking_schema import docking_input_schema
from chemgraph.tools.docking_core import (
    mock_docking,
    resolve_candidate_smiles,
    run_docking_core,
)

__all__ = [
    "mock_docking",
    "resolve_candidate_smiles",
    "run_docking",
    "run_docking_core",
]


@tool
def run_docking(docking_input: docking_input_schema) -> dict:
    """Dock a candidate molecule into a receptor and predict its binding affinity.

    Use this to estimate how strongly a small molecule binds a target and to obtain
    its best pose. The candidate may be a SMILES string, a molecule name, or a
    PubChem CID. The receptor is either a path to a prepared rigid receptor
    ``.pdbqt`` file, or a SMILES/name/CID for a small-molecule receptor.

    The search box is chosen automatically (``site_detection='auto'``: a supplied
    reference ligand, else fpocket if installed, else the whole receptor); the user
    may override it with ``center``/``box_size``. If the candidate, receptor, or
    ``n_poses`` are unspecified or ambiguous, ask the user rather than guessing.

    Parameters
    ----------
    docking_input : docking_input_schema
        Candidate, receptor, number of poses, site-detection method, and optional
        box override / reference ligand.

    Returns
    -------
    dict
        Resolved candidate, receptor, engine, chosen box, best binding affinity in
        kcal/mol (more negative = stronger), a per-pose list, and the poses file path.
    """
    return run_docking_core(docking_input)
