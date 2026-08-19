"""Compatibility tests for the legacy gRASPA interface."""

from unittest.mock import patch

from chemgraph.schemas.graspa_schema import graspa_input_schema
from chemgraph.tools.graspa_core import mock_graspa, run_graspa_core


def test_legacy_request_maps_to_single_component() -> None:
    params = graspa_input_schema(
        input_structure_file="framework.cif",
        adsorbate="N2",
        temperature=300,
        pressure=200000,
    )
    request = params.to_adsorption_request()
    assert request.components[0].name == "N2"
    assert request.components[0].mole_fraction == 1.0


@patch("chemgraph.tools.graspa_core.run_adsorption_core")
def test_legacy_result_exposes_scalar_uptake(mock_run) -> None:
    mock_run.return_value = {
        "status": "success",
        "components": [
            {
                "name": "CO2",
                "feed_mole_fraction": 1.0,
                "uptake": 1.5,
                "uncertainty": 0.1,
                "unit": "mol/kg",
            }
        ],
        "stdout_path": "/tmp/raspa.log",
    }
    params = graspa_input_schema(
        input_structure_file="framework.cif",
        adsorbate="CO2",
    )
    result = run_graspa_core(params, runtime={})
    assert result["uptake_in_mol_kg"] == 1.5
    assert result["output_result_file"] == "/tmp/raspa.log"


def test_mock_uses_existing_schema_field() -> None:
    params = graspa_input_schema(
        input_structure_file="framework.cif",
        adsorbate="H2O",
    )
    assert mock_graspa(params)["adsorbate"] == "H2O"
