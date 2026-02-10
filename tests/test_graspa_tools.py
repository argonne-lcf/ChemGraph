import pytest
from unittest.mock import patch, MagicMock
import shutil

from chemgraph.tools.graspa_tools import run_graspa_core, graspa_input_schema

@pytest.fixture
def mock_cif(tmp_path):
    """Creates a dummy CIF file for testing."""
    cif_file = tmp_path / "test_mof.cif"
    # Minimal CIF content for ase_read to not crash
    cif_file.write_text("data_test\n_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 10\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nC C 0 0 0")
    return cif_file

@patch("subprocess.run")
@patch("chemgraph.tools.graspa_tools._read_graspa_sycl_output")
def test_run_graspa_core_execution(mock_parser, mock_subproc, mock_cif):
    params = graspa_input_schema(
        input_structure_file=str(mock_cif),
        adsorbate="CO2",
        temperature=298.0,
        pressure=100000.0,
        n_cycles=100,
        output_result_file="raspa.log"
    )

    mock_subproc.return_value = MagicMock(returncode=0)
    mock_parser.return_value = {"status": "success", "uptake_in_mol_kg": 1.5}
    result = run_graspa_core(params)

    expected_dir_name = "test_mof--CO2-298.0-100000"
    sim_dir = mock_cif.parent / expected_dir_name
    assert sim_dir.exists()
    
    # Check if simulation.input was generated
    assert (sim_dir / "simulation.input").exists()
    
    # Check if subprocess was called with correct directory
    mock_subproc.assert_called_once()
    assert mock_subproc.call_args[1]['cwd'] == sim_dir
    
    # Check final output
    assert result["uptake_in_mol_kg"] == 1.5

    # Cleanup (optional with tmp_path)
    shutil.rmtree(sim_dir)