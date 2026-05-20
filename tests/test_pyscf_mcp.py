"""Tests for the first-iteration PySCF MCP tools."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

try:
    from fastmcp import Client

    from chemgraph.mcp.mcp_tools import mcp
    from chemgraph.schemas.pyscf_schema import PySCFMoleculeSpec
    from chemgraph.tools import pyscf_tools
    from chemgraph.tools.pyscf_tools import (
        _apply_pyscf_device,
        create_pyscf_molecule_core,
    )
except ModuleNotFoundError:
    pytest.skip("MCP test dependencies are not installed", allow_module_level=True)


def _pyscf_installed() -> bool:
    return importlib.util.find_spec("pyscf") is not None


@pytest.fixture
def h2_xyz(tmp_path):
    structure_file = tmp_path / "h2.xyz"
    structure_file.write_text(
        "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n",
        encoding="utf-8",
    )
    return structure_file


@pytest.fixture
def h2_crystal_xyz(tmp_path):
    structure_file = tmp_path / "h2_cell.xyz"
    structure_file.write_text(
        (
            '2\nLattice="5 0 0 0 5 0 0 0 5" '
            'Properties=species:S:1:pos:R:3 pbc="T T T"\n'
            "H 0.0 0.0 0.0\nH 0.0 0.0 0.74\n"
        ),
        encoding="utf-8",
    )
    return structure_file


@pytest.mark.asyncio
async def test_pyscf_mcp_exposes_only_first_iteration_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()

    pyscf_tool_names = {tool.name for tool in tools if "pyscf" in tool.name}
    assert pyscf_tool_names == {
        "create_pyscf_molecule",
        "create_pyscf_crystal",
        "run_pyscf_molecule",
        "run_pyscf_crystal",
    }


def test_pyscf_molecule_spec_defaults_to_cpu_device():
    spec = PySCFMoleculeSpec(
        source_structure_file="/tmp/h2.xyz",
        symbols=["H", "H"],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
    )

    assert spec.device == "cpu"


def test_pyscf_gpu_requires_gpu4pyscf_when_missing(monkeypatch):
    original_find_spec = pyscf_tools.importlib.util.find_spec

    def fake_find_spec(name):
        if name == "gpu4pyscf":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(pyscf_tools.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(ImportError, match="gpu4pyscf"):
        _apply_pyscf_device(object(), "gpu", "test object")


@pytest.mark.skipif(not _pyscf_installed(), reason="PySCF is not installed")
def test_create_pyscf_molecule_core(h2_xyz):
    result = create_pyscf_molecule_core(str(h2_xyz), basis="sto-3g")

    assert result["status"] == "success"
    assert result["object_type"] == "pyscf_molecule"
    assert result["pyscf_molecule"]["device"] == "cpu"
    assert result["molecule"]["natoms"] == 2


@pytest.mark.skipif(not _pyscf_installed(), reason="PySCF is not installed")
@pytest.mark.asyncio
async def test_create_and_run_pyscf_molecule_energy(h2_xyz, tmp_path):
    async with Client(mcp) as client:
        created = await client.call_tool(
            "create_pyscf_molecule",
            {"structure_file": str(h2_xyz), "basis": "sto-3g"},
        )
        molecule_payload = json.loads(created.content[0].text)

        run = await client.call_tool(
            "run_pyscf_molecule",
            {
                "pyscf_molecule": molecule_payload["pyscf_molecule"],
                "driver": "energy",
                "output_json": str(tmp_path / "pyscf_molecule_results.json"),
            },
        )

    payload = json.loads(run.content[0].text)
    assert payload["status"] == "success"
    assert payload["driver"] == "energy"
    assert payload["energy"]["hartree"] < 0
    assert payload["scf"]["converged"] is True
    assert Path(payload["artifacts"]["output_json"]).exists()


@pytest.mark.skipif(not _pyscf_installed(), reason="PySCF is not installed")
@pytest.mark.asyncio
async def test_run_pyscf_molecule_loads_create_output_json(h2_xyz, tmp_path):
    molecule_json = tmp_path / "h2_pyscf_molecule.json"
    results_json = tmp_path / "h2_pyscf_results.json"

    async with Client(mcp) as client:
        created = await client.call_tool(
            "create_pyscf_molecule",
            {
                "structure_file": str(h2_xyz),
                "basis": "sto-3g",
                "output_json": str(molecule_json),
            },
        )
        created_payload = json.loads(created.content[0].text)

        run = await client.call_tool(
            "run_pyscf_molecule",
            {
                "pyscf_molecule_json": str(molecule_json),
                "driver": "energy",
                "output_json": str(results_json),
            },
        )

    payload = json.loads(run.content[0].text)
    assert created_payload["artifacts"]["output_json"] == str(molecule_json.resolve())
    assert payload["status"] == "success"
    assert payload["driver"] == "energy"
    assert payload["input"]["pyscf_molecule_json"] == str(molecule_json.resolve())
    assert payload["energy"]["hartree"] < 0
    assert Path(payload["artifacts"]["output_json"]).exists()


@pytest.mark.skipif(not _pyscf_installed(), reason="PySCF is not installed")
@pytest.mark.asyncio
async def test_create_and_run_pyscf_crystal_energy(h2_crystal_xyz, tmp_path):
    async with Client(mcp) as client:
        created = await client.call_tool(
            "create_pyscf_crystal",
            {
                "structure_file": str(h2_crystal_xyz),
                "basis": "gth-szv",
                "pseudo": "gth-pade",
            },
        )
        crystal_payload = json.loads(created.content[0].text)

        run = await client.call_tool(
            "run_pyscf_crystal",
            {
                "pyscf_crystal": crystal_payload["pyscf_crystal"],
                "driver": "energy",
                "output_json": str(tmp_path / "pyscf_crystal_results.json"),
            },
        )

    payload = json.loads(run.content[0].text)
    assert payload["status"] == "success"
    assert payload["driver"] == "energy"
    assert payload["energy"]["hartree"] < 0
    assert payload["scf"]["converged"] is True
    assert Path(payload["artifacts"]["output_json"]).exists()


@pytest.mark.skipif(not _pyscf_installed(), reason="PySCF is not installed")
@pytest.mark.asyncio
async def test_pyscf_crystal_thermochemistry_reports_first_iteration_limit(
    h2_crystal_xyz,
):
    async with Client(mcp) as client:
        created = await client.call_tool(
            "create_pyscf_crystal",
            {"structure_file": str(h2_crystal_xyz)},
        )
        crystal_payload = json.loads(created.content[0].text)

        run = await client.call_tool(
            "run_pyscf_crystal",
            {
                "pyscf_crystal": crystal_payload["pyscf_crystal"],
                "driver": "thermochemistry",
                "output_json": None,
            },
        )

    payload = json.loads(run.content[0].text)
    assert payload["status"] == "failure"
    assert payload["error_type"] == "NotImplementedError"
