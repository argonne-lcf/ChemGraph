"""Tests for PySCF MCP wrappers.

The executable PySCF tests are skipped when PySCF is not installed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

try:
    from fastmcp import Client
    from chemgraph.mcp.mcp_tools import mcp
    from chemgraph.schemas.pyscf_schema import PySCFPropertyInput
    from chemgraph.tools.pyscf_tools import run_pyscf_property_core
except ModuleNotFoundError:
    pytest.skip("MCP test dependencies are not installed", allow_module_level=True)


def _pyscf_installed() -> bool:
    return importlib.util.find_spec("pyscf") is not None


@pytest.mark.asyncio
async def test_get_pyscf_capability_manifest():
    async with Client(mcp) as client:
        res = await client.call_tool("get_pyscf_capability_manifest", {})

    payload = json.loads(res.content[0].text)
    assert payload["status"] == "success"
    assert payload["manifest"] == "pyscf_capability_manifest"
    assert "run_pyscf_molecular" in payload["tools"]
    assert "casscf_single_point" in payload["tools"]["run_pyscf_recipe"]["recipes"]


@pytest.mark.asyncio
async def test_extract_pyscf_output(tmp_path):
    result_file = tmp_path / "pyscf_result.json"
    result_file.write_text(
        json.dumps(
            {
                "status": "success",
                "calculation": "pyscf_molecular",
                "properties": {"dipole": {"value": [0.0, 0.0, 0.0]}},
            }
        ),
        encoding="utf-8",
    )

    async with Client(mcp) as client:
        res = await client.call_tool(
            "extract_pyscf_output", {"json_file": str(result_file)}
        )

    payload = json.loads(res.content[0].text)
    assert payload["status"] == "success"
    assert payload["calculation"] == "pyscf_molecular"


@pytest.mark.asyncio
async def test_run_pyscf_property_extracts_stored_properties(tmp_path):
    result_file = tmp_path / "pyscf_result.json"
    result_file.write_text(
        json.dumps(
            {
                "status": "success",
                "properties": {
                    "dipole": {"value": [0.0, 0.0, 1.0], "unit": "Debye"},
                    "mo_energy": {"value": [-0.5], "unit": "Hartree"},
                },
            }
        ),
        encoding="utf-8",
    )

    async with Client(mcp) as client:
        res = await client.call_tool(
            "run_pyscf_property",
            {
                "params": {
                    "result_json": str(result_file),
                    "properties": ["dipole"],
                }
            },
        )

    payload = json.loads(res.content[0].text)
    assert payload["status"] == "success"
    assert payload["properties"]["dipole"]["unit"] == "Debye"


def test_run_pyscf_property_raises_for_missing_requested_property(tmp_path):
    result_file = tmp_path / "pyscf_result.json"
    result_file.write_text(
        json.dumps({"status": "success", "properties": {}}),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="Requested properties were not found"):
        run_pyscf_property_core(
            PySCFPropertyInput(
                result_json=str(result_file),
                properties=["dipole"],
            )
        )


@pytest.mark.skipif(not _pyscf_installed(), reason="PySCF is not installed")
@pytest.mark.asyncio
async def test_run_pyscf_molecular_h2_hf_sto3g(tmp_path):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text(
        "2\nH2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n",
        encoding="utf-8",
    )

    async with Client(mcp) as client:
        res = await client.call_tool(
            "run_pyscf_molecular",
            {
                "params": {
                    "structure": {"input_structure_file": str(xyz)},
                    "basis": "sto-3g",
                    "reference": "RHF",
                    "properties": ["dipole", "mo_energy"],
                    "output_dir": str(tmp_path / "pyscf"),
                }
            },
        )

    payload = json.loads(res.content[0].text)
    assert payload["status"] == "success"
    assert payload["scf"]["reference"] == "RHF"
    assert payload["scf"]["converged"] is True
    assert payload["scf"]["total_energy"]["hartree"] < 0
    assert Path(payload["artifacts"]["output_json"]).exists()
