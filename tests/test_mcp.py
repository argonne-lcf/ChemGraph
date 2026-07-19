"""Test suite for MCP servers."""

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

try:
    from mcp.types import TextContent
    from fastmcp import Client
    from chemgraph.mcp.cg_fastmcp import CGFastMCP
    from chemgraph.mcp.mcp_tools import mcp
    from chemgraph.mcp.data_analysis_mcp import mcp as data_mcp
except ModuleNotFoundError:
    pytest.skip("MCP test dependencies are not installed", allow_module_level=True)

TEST_DIR = Path(__file__).parent


def _fanout_worker(item: dict) -> dict:
    return item


def test_schema_fanout_tool_advertises_batch_result_signature(monkeypatch):
    """Fanout tools expose an ensemble input but return batch summaries."""
    local_mcp = CGFastMCP(name="test")
    captured = {}

    def capture_tool(fn, **kwargs):
        captured["fn"] = fn
        captured["kwargs"] = kwargs

    monkeypatch.setattr(local_mcp, "add_tool", capture_tool)

    @local_mcp.schema_fanout_tool(name="fanout", worker=_fanout_worker)
    def fanout(params: dict) -> list[dict]:
        return [params]

    sig = inspect.signature(captured["fn"])

    assert list(sig.parameters) == ["params"]
    assert sig.parameters["params"].annotation is dict
    assert sig.return_annotation == dict[str, Any]


def test_mace_worker_creates_inline_output_parent(monkeypatch):
    from ase import Atoms

    from chemgraph.mcp import mace_mcp_hpc
    from chemgraph.tools.ase_core import atoms_to_atomsdata

    atoms = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 0.74]])
    output_file = "nested/family/output.json"

    def fake_run_mace_core(params):
        output_path = Path(params.output_result_file)
        assert output_path.parent.is_dir()
        output_path.write_text('{"ok": true}', encoding="utf-8")
        return {"status": "success"}

    monkeypatch.setattr(mace_mcp_hpc, "run_mace_core", fake_run_mace_core)

    result = mace_mcp_hpc._mace_worker(
        {
            "inline_structure": atoms_to_atomsdata(atoms).model_dump(),
            "output_result_file": output_file,
            "driver": "energy",
            "model": "small",
            "device": "cpu",
        }
    )

    # The worker returns run_mace_core's result verbatim; full_output read-back
    # was intentionally dropped. The inline output parent dir is asserted inside
    # fake_run_mace_core above.
    assert result["status"] == "success"


def test_run_ase_core_creates_output_parent_directory(monkeypatch, tmp_path):
    """run_ase_core should mkdir the output file's parent before writing.

    Academy agents and CLI users routinely point output_results_file at a
    not-yet-existing nested subdirectory of a shared run dir. Without this,
    the final ``open(output_results_file, "w")`` fails with
    FileNotFoundError after the calculation has already burned its compute
    time.
    """
    from ase import Atoms
    from ase.io import write as ase_write

    from chemgraph.schemas.ase_input import ASEInputSchema
    from chemgraph.tools import ase_core

    # Real XYZ that ase.io.read can parse.
    input_path = tmp_path / "h2.xyz"
    ase_write(input_path, Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 0.74]]))

    # Output path under a nested subdirectory that does NOT exist yet.
    output_path = tmp_path / "deeply" / "nested" / "output.json"
    assert not output_path.parent.exists()

    class _FakeCalc:
        # ASE's Atoms.get_potential_energy invokes self._calc.get_potential_energy(atoms).
        def get_potential_energy(self, _atoms=None, force_consistent=False):
            return -1.234

    def fake_load_calculator(_calculator):
        return _FakeCalc(), {}, None

    monkeypatch.setattr(ase_core, "load_calculator", fake_load_calculator)

    params = ASEInputSchema(
        input_structure_file=str(input_path),
        output_results_file=str(output_path),
        driver="energy",
        calculator={"calculator_type": "emt"},
    )

    result = ase_core.run_ase_core(params)

    assert result["status"] == "success", result
    assert output_path.exists()
    assert output_path.parent.is_dir()


@pytest.mark.asyncio
async def test_split_cif_dataset(tmp_path):
    """Test splitting a dataset of CIF files."""
    # Create dummy CIF files
    input_dir = tmp_path / "input_cifs"
    input_dir.mkdir()
    for i in range(5):
        (input_dir / f"mof_{i}.cif").touch()

    output_root = tmp_path / "batches"

    async with Client(data_mcp) as client:
        # Test split by batch_size
        res = await client.call_tool(
            "split_cif_dataset",
            {
                "input_dir": str(input_dir),
                "output_root": str(output_root),
                "batch_size": 2,
            },
        )
        assert "Success" in res.content[0].text
        assert "batches" in res.content[0].text

        # Verify output structure
        assert (output_root / "batch_000").exists()
        assert (output_root / "batch_001").exists()
        assert (output_root / "batch_002").exists()  # 5 files / 2 = 3 batches
        assert len(list((output_root / "batch_000").iterdir())) == 2


@pytest.fixture(name="base_ase_input_dict")
def fixture_base_ase_input_dict():
    """Fixture providing base ASE input parameters."""
    return {
        "input_structure_file": str(TEST_DIR / "water.xyz"),
        "output_results_file": str(TEST_DIR / "water_output.json"),
        "optimizer": "bfgs",
        "calculator": {
            "calculator_type": "mace_mp",
        },
    }


@pytest.mark.asyncio
async def test_run_ase_energy_simple(base_ase_input_dict):
    """Simplified test using FastMCP's in-memory client."""

    input_data = base_ase_input_dict.copy()
    input_data["driver"] = "energy"

    async with Client(mcp) as client:
        res = await client.call_tool("run_ase", {"params": input_data})
        result_dict = json.loads(res.content[0].text)
        assert result_dict["status"] == "success"
        assert "single_point_energy" in result_dict

        assert isinstance(res.content[0], TextContent)
        assert "Simulation completed" in res.content[0].text


@pytest.mark.asyncio
async def test_run_ase_thermo_simple(base_ase_input_dict):
    """Simplified thermochemistry test"""
    input_data = base_ase_input_dict.copy()
    input_data["driver"] = "thermo"
    input_data["temperature"] = 298.15

    async with Client(mcp) as client:
        res = await client.call_tool("run_ase", {"params": input_data})
        result_dict = json.loads(res.content[0].text)

        assert result_dict["status"] == "success"
        assert "result" in result_dict

        assert isinstance(res.content[0], TextContent)
        assert "Thermochemistry computed and returned" in res.content[0].text


@pytest.mark.asyncio
async def test_aggregate_and_rank(tmp_path):
    """Test aggregating simulation results and ranking them."""
    # Create dummy JSONL output
    log_file = tmp_path / "simulations.jsonl"
    dummy_data = [
        {
            "status": "success",
            "cif_path": "/abs/path/to/mof_1.cif",
            "uptake_in_mol_kg": 10.5,
            "temperature_in_K": 298.0,
            "pressure_in_Pa": 100000.0,
        },
        {
            "status": "success",
            "cif_path": "/abs/path/to/mof_2.cif",
            "uptake_in_mol_kg": 2.1,
            "temperature_in_K": 298.15,
            "pressure_in_Pa": 100000.0,
        },
        {"status": "failure", "cif_path": "/abs/path/to/mof_3.cif"},
    ]

    with open(log_file, "w", encoding="utf-8") as f:
        for entry in dummy_data:
            f.write(json.dumps(entry) + "\n")

    output_csv = tmp_path / "results.csv"

    async with Client(data_mcp) as client:
        # 1. Aggregate
        res_agg = await client.call_tool(
            "aggregate_simulation_results",
            {"file_paths": [str(log_file)], "output_csv_path": str(output_csv)},
        )
        assert "Success" in res_agg.content[0].text
        assert output_csv.exists()

        # 2. Ranking
        # Note: We simulate a ranking query.
        res_rank = await client.call_tool(
            "rank_mofs_performance",
            {
                "input_csv_path": str(output_csv),
                "ads_pressure": 100000.0,
                "ads_temp": 298.0,
                "top_percentile": 1.0,  # Return all
            },
        )
        text = res_rank.content[0].text
        assert "Analysis Complete" in text
        assert "mof_1.cif" in text
        assert "mof_2.cif" in text  # Should find both due to tolerance


# ---------------------------------------------------------------------------
# Bare-name path resolution across MCP tools (small-model safety net).
# A file written by a sibling tool lands in CHEMGRAPH_LOG_DIR; a small model
# may echo back just the bare name. These tools must resolve it there.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_run_ase_resolves_bare_name_from_log_dir(monkeypatch, tmp_path):
    """The general MCP run_ase resolves a bare input name via the log dir."""
    from ase.build import molecule
    from ase.io import write as ase_write

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    ase_write(str(tmp_path / "h2o.xyz"), molecule("H2O"))

    input_data = {
        "input_structure_file": "h2o.xyz",  # bare name, not an absolute path
        "output_results_file": "h2o_energy.json",
        "driver": "energy",
        "calculator": {"calculator_type": "emt"},
    }
    async with Client(mcp) as client:
        res = await client.call_tool("run_ase", {"params": input_data})
        result_dict = json.loads(res.content[0].text)
        assert result_dict["status"] == "success"


@pytest.mark.asyncio
async def test_mcp_extract_output_json_resolves_bare_name(monkeypatch, tmp_path):
    """extract_output_json resolves a bare JSON name via the log dir."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    (tmp_path / "result.json").write_text('{"ok": true}', encoding="utf-8")

    async with Client(mcp) as client:
        res = await client.call_tool("extract_output_json", {"json_file": "result.json"})
        payload = json.loads(res.content[0].text)
        assert payload["ok"] is True


def test_embed_inline_resolves_bare_name(monkeypatch, tmp_path):
    """_embed_inline_if_local resolves a bare name and embeds the structure."""
    from ase.build import molecule
    from ase.io import write as ase_write

    from chemgraph.mcp import mace_mcp_hpc

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    ase_write(str(tmp_path / "h2o.xyz"), molecule("H2O"))

    job = {"input_structure_file": "h2o.xyz"}  # bare name
    mace_mcp_hpc._embed_inline_if_local(job)

    assert job["input_structure_file"] == str(tmp_path / "h2o.xyz")
    assert "inline_structure" in job


def test_embed_inline_leaves_remote_and_missing_alone(monkeypatch, tmp_path):
    """Remote paths and genuinely missing files are deferred to the worker."""
    from chemgraph.mcp import mace_mcp_hpc

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))

    remote_job = {
        "input_structure_file": "x.xyz",
        "remote_structure_file": "/remote/x.xyz",
    }
    mace_mcp_hpc._embed_inline_if_local(remote_job)
    assert "inline_structure" not in remote_job

    missing_job = {"input_structure_file": "not_here.xyz"}
    mace_mcp_hpc._embed_inline_if_local(missing_job)
    assert "inline_structure" not in missing_job


def test_data_analysis_aggregate_resolves_bare_name(monkeypatch, tmp_path):
    """aggregate_simulation_results reads a bare-name file from the log dir."""
    from chemgraph.mcp import data_analysis_mcp

    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    (tmp_path / "sim.jsonl").write_text(
        json.dumps(
            {
                "status": "success",
                "cif_path": "/abs/mof_1.cif",
                "uptake_in_mol_kg": 5.0,
                "temperature_in_K": 298.0,
                "pressure_in_Pa": 1e5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    out_csv = tmp_path / "agg.csv"

    # Tool function with a bare input name (resolved against the log dir).
    msg = data_analysis_mcp.aggregate_simulation_results(
        file_paths=["sim.jsonl"],
        output_csv_path=str(out_csv),
    )
    assert "Success" in msg
    assert out_csv.exists()
