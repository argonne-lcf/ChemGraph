"""LangChain ``@tool`` wrapper for molecular docking.

Delegates to the pure-Python implementation in
:mod:`chemgraph.tools.docking_core`.
"""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.docking_schema import docking_input_schema
from chemgraph.tools.docking_core import (
    # Re-export core helpers for convenience / parity with other tool modules.
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
    PubChem CID (it is resolved automatically). The receptor defaults to
    ``"vancomycin"`` (a bundled target) or may be a path to a prepared rigid
    receptor ``.pdbqt`` file, in which case ``center`` and ``box_size`` are required.

    If the user has not specified ``n_poses``, ask them or use the default (10).

    Parameters
    ----------
    docking_input : docking_input_schema
        Candidate, receptor, number of poses, optional search box, and exhaustiveness.

    Returns
    -------
    dict
        Resolved candidate, receptor, engine, best binding affinity in kcal/mol
        (more negative = stronger), a per-pose list, and the poses file path.
    """
    return run_docking_core(docking_input)
