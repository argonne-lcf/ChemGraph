import sys
from unittest.mock import MagicMock, patch

import pytest

from chemgraph.tools.pacmof2_tools import (
    mock_pacmof2,
    run_pacmof2_core,
)
from chemgraph.schemas.pacmof2_schema import pacmof2_input_schema


@pytest.fixture
def mock_cif(tmp_path):
    """Creates a dummy CIF file for testing."""
    cif_file = tmp_path / "test_mof.cif"
    # Minimal CIF content for ase_read to not crash.
    cif_file.write_text(
        "data_test\n_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 10\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_atom_site_label\n_atom_site_type_symbol\n"
        "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
        "C C 0 0 0"
    )
    return cif_file


def test_run_pacmof2_core_execution(mock_cif):
    """run_pacmof2_core should call get_charges with resolved args and parse output."""
    params = pacmof2_input_schema(
        input_structure_file=str(mock_cif),
        identifier="_pacmof",
        adjust_charge_method="mean",
        net_charge=0,
    )

    # Inject a fake pacmof2 module so the lazy `from pacmof2 import get_charges`
    # inside run_pacmof2_core resolves without the real package installed.
    fake_pacmof2 = MagicMock()
    expected_summary = {"status": "success", "output_cif_path": "x", "sum_of_charges": 0.0}

    with patch.dict(sys.modules, {"pacmof2": fake_pacmof2}), patch(
        "chemgraph.tools.pacmof2_core._read_pacmof2_output"
    ) as mock_parser:
        mock_parser.return_value = expected_summary
        result = run_pacmof2_core(params)

    # get_charges called once with the resolved path and passthrough params.
    fake_pacmof2.get_charges.assert_called_once()
    call_kwargs = fake_pacmof2.get_charges.call_args.kwargs
    assert call_kwargs["path_to_cif"] == str(mock_cif.resolve())
    assert call_kwargs["identifier"] == "_pacmof"
    assert call_kwargs["adjust_charge_method"] == "mean"
    assert call_kwargs["net_charge"] == 0
    assert call_kwargs["multiple_cifs"] is False

    # Parser receives the derived output CIF path.
    parser_kwargs = mock_parser.call_args.kwargs
    assert parser_kwargs["output_cif"].endswith("test_mof_pacmof.cif")
    assert result == expected_summary


def test_net_charge_dict_passthrough(mock_cif):
    """An ionic-MOF net_charge dict should reach get_charges unchanged."""
    net = {"O": -0.5}
    params = pacmof2_input_schema(
        input_structure_file=str(mock_cif), net_charge=net
    )
    fake_pacmof2 = MagicMock()
    with patch.dict(sys.modules, {"pacmof2": fake_pacmof2}), patch(
        "chemgraph.tools.pacmof2_core._read_pacmof2_output",
        return_value={"status": "success"},
    ):
        run_pacmof2_core(params)
    assert fake_pacmof2.get_charges.call_args.kwargs["net_charge"] == net


def test_mock_pacmof2(mock_cif):
    """mock_pacmof2 returns a plausible summary without pacmof2 installed."""
    params = pacmof2_input_schema(input_structure_file=str(mock_cif))
    result = mock_pacmof2(params)
    assert result["status"] == "success"
    assert result["output_cif_path"].endswith("test_mof_pacmof.cif")
    assert isinstance(result["per_element_mean_charge"], dict)


def test_missing_cif_raises():
    """A non-existent input CIF should raise FileNotFoundError."""
    params = pacmof2_input_schema(input_structure_file="/no/such/file.cif")
    with pytest.raises(FileNotFoundError):
        run_pacmof2_core(params)


def test_import_error_message(mock_cif):
    """When pacmof2 is not importable, a clear RuntimeError is raised."""
    params = pacmof2_input_schema(input_structure_file=str(mock_cif))
    # Ensure importing pacmof2 fails even if it happens to be installed.
    with patch.dict(sys.modules, {"pacmof2": None}):
        with pytest.raises(RuntimeError, match="PACMOF2 is not installed"):
            run_pacmof2_core(params)
