"""Tests for the molecular docking tool.

Hermetic by default: candidate resolution uses a SMILES (no network), docking is
exercised via ``mock_docking`` and via a graph-construction check. The real Vina
path is only run when the optional ``docking`` extra (vina + meeko) is installed.
"""

from unittest.mock import MagicMock

import pytest

from chemgraph.schemas.docking_schema import docking_input_schema
from chemgraph.tools.docking_core import mock_docking, resolve_candidate_smiles


def test_resolve_candidate_smiles_passthrough():
    """A valid SMILES is canonicalized without any network lookup."""
    out = resolve_candidate_smiles("OC(=O)C")  # acetic acid, non-canonical order
    from rdkit import Chem

    assert out == Chem.CanonSmiles("CC(=O)O")


def test_mock_docking_shape():
    """mock_docking returns the expected structure and pose count."""
    params = docking_input_schema(candidate="CC(=O)O", n_poses=4)
    res = mock_docking(params)
    assert res["engine"] == "mock"
    assert res["n_poses"] == 4
    assert len(res["poses"]) == 4
    assert isinstance(res["best_affinity_kcal_mol"], float)
    assert res["receptor"] == "vancomycin"


def test_construct_docking_graph_builds():
    """The docking graph compiles with a mock LLM (no API calls)."""
    from chemgraph.graphs.docking_agent import construct_docking_graph

    graph = construct_docking_graph(MagicMock())
    assert graph is not None


def test_run_docking_core_vina(monkeypatch, tmp_path):
    """Real Vina dock into the bundled vancomycin target (skipped without the extra)."""
    pytest.importorskip("vina")
    pytest.importorskip("meeko")
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))

    from chemgraph.tools.docking_core import run_docking_core

    params = docking_input_schema(
        candidate="CC(=O)O", receptor="vancomycin", n_poses=2, exhaustiveness=1
    )
    res = run_docking_core(params)
    assert res["engine"] == "vina"
    assert isinstance(res["best_affinity_kcal_mol"], float)
    assert res["n_poses"] >= 1
