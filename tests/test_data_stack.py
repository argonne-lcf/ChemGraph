"""Regression tests for ChemGraph's numerical and tabular data stack."""

import json

import numpy as np
import pandas as pd
import pytest
from ase import Atoms

from chemgraph.mcp.data_analysis_mcp import (
    aggregate_simulation_results,
    rank_mofs_performance,
)
from chemgraph.tools.ase_core import atoms_to_atomsdata, atomsdata_to_atoms
from chemgraph.tools.generic_tools import calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2 * 3", 7.0),
        ("2 * pi + e", 2 * np.pi + np.e),
        ("sin(pi / 2) + sqrt(9) + abs(-2)", 6.0),
    ],
)
def test_numexpr_calculator_supported_expressions(expression, expected):
    """The LangChain calculator tool should evaluate supported expressions."""
    result = calculator.invoke({"expression": expression})

    assert float(result) == pytest.approx(expected)


def test_numexpr_calculator_reports_invalid_and_empty_expressions():
    """Invalid calculator input should return an actionable error string."""
    assert calculator.invoke({"expression": "  "}) == "Error: Empty expression"
    assert calculator.invoke({"expression": "unknown_name + 1"}).startswith(
        "Error evaluating expression:"
    )


def test_numpy_ase_round_trip_is_json_serializable():
    """ASE's NumPy arrays should become JSON-safe AtomsData lists."""
    atoms = Atoms(
        numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.76, 0.58, 0.0],
                [-0.76, 0.58, 0.0],
            ]
        ),
        cell=np.eye(3),
        pbc=np.array([False, False, False]),
    )

    atoms_data = atoms_to_atomsdata(atoms)
    payload = atoms_data.model_dump()
    restored = atomsdata_to_atoms(atoms_data)

    assert isinstance(atoms_data.numbers, list)
    assert isinstance(atoms_data.positions, list)
    assert isinstance(atoms_data.cell, list)
    assert isinstance(atoms_data.pbc, list)
    json.dumps(payload)
    np.testing.assert_array_equal(restored.numbers, atoms.numbers)
    np.testing.assert_allclose(restored.positions, atoms.positions)
    np.testing.assert_allclose(restored.cell.array, atoms.cell.array)
    np.testing.assert_array_equal(restored.pbc, atoms.pbc)


def test_pandas_aggregation_and_ranking(monkeypatch, tmp_path):
    """JSONL aggregation should coerce numerics and rank valid MOF records."""
    monkeypatch.setenv("CHEMGRAPH_LOG_DIR", str(tmp_path))
    records = [
        {
            "status": "success",
            "cif_path": "/structures/low.cif",
            "uptake_in_mol_kg": "4.0",
            "temperature_in_K": "298.0",
            "pressure_in_Pa": "100000",
        },
        {
            "status": "success",
            "cif_path": "/structures/low.cif",
            "uptake_in_mol_kg": "1.0",
            "temperature_in_K": "300.0",
            "pressure_in_Pa": "10000",
        },
        {
            "status": "success",
            "cif_path": "/structures/high.cif",
            "uptake_in_mol_kg": "8.0",
            "temperature_in_K": "298.0",
            "pressure_in_Pa": "100000",
        },
        {
            "status": "success",
            "cif_path": "/structures/high.cif",
            "uptake_in_mol_kg": "2.0",
            "temperature_in_K": "300.0",
            "pressure_in_Pa": "10000",
        },
        {
            "status": "success",
            "cif_path": "/structures/incomplete.cif",
            "uptake_in_mol_kg": None,
            "temperature_in_K": "298.0",
            "pressure_in_Pa": "100000",
        },
        {
            "status": "failed",
            "cif_path": "/structures/ignored.cif",
            "uptake_in_mol_kg": 99,
            "temperature_in_K": 298,
            "pressure_in_Pa": 100000,
        },
    ]
    input_path = tmp_path / "results.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records)
        + "\n\n{malformed json}\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "aggregated.csv"

    result = aggregate_simulation_results(
        file_paths=["results.jsonl", "", str(tmp_path / "missing.jsonl")],
        output_csv_path=str(output_path),
    )

    assert "Success: Aggregated 5 records" in result
    aggregated = pd.read_csv(output_path)
    assert list(aggregated["cif_filename"]) == [
        "low.cif",
        "low.cif",
        "high.cif",
        "high.cif",
        "incomplete.cif",
    ]
    assert pd.api.types.is_numeric_dtype(aggregated["uptake_in_mol_kg"])
    assert pd.api.types.is_numeric_dtype(aggregated["temperature"])
    assert pd.api.types.is_numeric_dtype(aggregated["pressure"])
    assert pd.isna(
        aggregated.loc[
            aggregated["cif_filename"] == "incomplete.cif",
            "uptake_in_mol_kg",
        ].iloc[0]
    )

    absolute = rank_mofs_performance(
        input_csv_path=str(output_path),
        ads_pressure=100000,
        ads_temp=298,
        top_percentile=1.0,
    )
    assert "Found 2 candidates (out of 2 valid MOFs)" in absolute
    assert absolute.index("high.cif") < absolute.index("low.cif")
    assert "8.0" in absolute
    assert "4.0" in absolute

    working_capacity = rank_mofs_performance(
        input_csv_path=str(output_path),
        ads_pressure=100000,
        ads_temp=298,
        des_pressure=10000,
        des_temp=300,
        top_percentile=1.0,
    )
    assert "Found 2 candidates (out of 2 valid MOFs)" in working_capacity
    assert working_capacity.index("high.cif") < working_capacity.index("low.cif")
    assert "6.0" in working_capacity
    assert "3.0" in working_capacity


def test_pandas_aggregation_rejects_files_without_success_records(tmp_path):
    """Malformed, failed, and missing inputs should not produce an empty CSV."""
    input_path = tmp_path / "invalid.jsonl"
    input_path.write_text(
        "\n{malformed json}\n"
        + json.dumps({"status": "failed", "cif_path": "/structures/failed.cif"})
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "aggregated.csv"

    result = aggregate_simulation_results(
        file_paths=[str(input_path), str(tmp_path / "missing.jsonl")],
        output_csv_path=str(output_path),
    )

    assert result == "Error: No valid success data found in the provided file list."
    assert not output_path.exists()
