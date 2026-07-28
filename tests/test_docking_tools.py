"""Tests for the molecular docking tool.

Hermetic by default: candidate resolution uses a SMILES (no network), docking is
exercised via ``mock_docking`` and by binding the tool to the single-agent graph.
The real Vina path runs only when the optional ``docking`` extra (vina + meeko) is
installed.
"""

from unittest.mock import MagicMock

import pytest

from chemgraph.schemas.docking_schema import docking_input_schema
from chemgraph.tools.docking_core import mock_docking, resolve_candidate_smiles


def test_resolve_candidate_smiles_passthrough():
    """A valid SMILES is canonicalized without any network lookup."""
    from rdkit import Chem

    assert resolve_candidate_smiles("OC(=O)C") == Chem.CanonSmiles("CC(=O)O")


def test_mock_docking_shape():
    """mock_docking returns the expected structure and pose count."""
    params = docking_input_schema(candidate="CC(=O)O", receptor="benzene", n_poses=4)
    res = mock_docking(params)
    assert res["engine"] == "mock"
    assert res["n_poses"] == 4
    assert len(res["poses"]) == 4
    assert isinstance(res["best_affinity_kcal_mol"], float)


def test_single_agent_binds_run_docking():
    """The single-agent graph builds with run_docking bound (no LLM calls)."""
    from chemgraph.graphs.single_agent import construct_single_agent_graph
    from chemgraph.tools.docking_tools import run_docking

    graph = construct_single_agent_graph(MagicMock(), tools=[run_docking])
    assert graph is not None


def test_run_docking_core_vina(monkeypatch, tmp_path):
    """Real Vina dock into a small SMILES receptor (skipped without the extra)."""
    pytest.importorskip("vina")
    pytest.importorskip("meeko")
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))

    from chemgraph.tools.docking_core import run_docking_core

    params = docking_input_schema(
        candidate="CCO",
        receptor="c1ccccc1",  # benzene, as a small-molecule receptor
        site_detection="blind",
        n_poses=2,
        exhaustiveness=1,
    )
    res = run_docking_core(params)
    assert res["engine"] == "vina"
    assert isinstance(res["best_affinity_kcal_mol"], float)
    assert res["n_poses"] >= 1
