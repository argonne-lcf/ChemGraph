"""Workspace and configuration checks without the optional HTTP dependencies."""

import pytest

from chemgraph.api.settings import Settings

def test_workspace_tools_enforce_paths_calculators_and_emt(tmp_path, monkeypatch):
    from chemgraph.api.chemistry import web_tools, workspace_path, structure_xyz

    workspace = tmp_path / "conversation"
    turn = workspace / "turn"
    turn.mkdir(parents=True)
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(turn))
    monkeypatch.chdir(turn)
    outside = tmp_path / "private.json"
    outside.write_text('{"secret": "value"}')
    with pytest.raises(ValueError):
        workspace_path(workspace, str(outside), reading=True)
    with pytest.raises(ValueError):
        workspace_path(workspace, "../old-result.json")
    tools = {tool.name: tool for tool in web_tools(workspace, ("emt",))}
    assert set(tools) == {
        "molecule_name_to_smiles",
        "smiles_to_coordinate_file",
        "run_ase",
        "extract_output_json",
        "calculator",
        "read_workspace_file",
    }
    (turn / "copper.xyz").write_text("2\nCu dimer\nCu 0 0 0\nCu 0 0 2.5\n")
    params = {
        "input_structure_file": "copper.xyz",
        "output_results_file": "result.json",
        "driver": "energy",
        "calculator": {"calculator_type": "emt"},
    }
    result = tools["run_ase"].invoke({"ase_input": params})
    assert result["potential_energy"] > 0
    assert "Cu" in structure_xyz(turn / "copper.xyz")
    assert (turn / "result.json").exists()
    with pytest.raises(ValueError):
        tools["extract_output_json"].invoke({"json_file": str(outside)})
    with pytest.raises(ValueError):
        tools["run_ase"].invoke(
            {"ase_input": {**params, "calculator": {"calculator_type": "mace_mp"}}}
        )


def test_settings_require_explicit_demo_user_and_approved_calculators():
    with pytest.raises(ValueError):
        Settings(demo=True)
    with pytest.raises(ValueError):
        Settings(calculators=("orca",))
