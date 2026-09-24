import csv
import math

import pytest

from chemgraph.schemas.graspa_analysis import GraspaAnalysis
from chemgraph.tools.graspa_analysis import analyze_records, rank_records, normalize_record


def row(source="/one/same.cif", pressure=960, uptake=8, **extra):
    return {"input_structure_file": source, "temperature_in_K": 298,
            "pressure_in_Pa": pressure, "uptake_in_mol_kg": uptake, "status": "success", **extra}


def spec(**extra):
    return GraspaAnalysis(adsorption={"temperature": 298, "pressure": 960},
                          desorption={"temperature": 298, "pressure": 320}, **extra)


def test_full_source_identity_repeats_and_failed_pairs(tmp_path):
    rows = [row(), row(uptake=10), row(pressure=320, uptake=3), row(pressure=320, uptake=5),
            row("/two/same.cif"), row("/two/same.cif", pressure=320, status="failure"),
            row("/three/same.cif")]
    ranked, excluded = rank_records(rows, spec())
    assert len(ranked) == 1
    assert ranked[0]["working_capacity"] == 5
    assert ranked[0] == {"input_structure_file": "/one/same.cif", "uptake_ads": 9,
                         "uptake_des": 4, "working_capacity": 5}
    assert {r["input_structure_file"] for r in excluded} == {"/two/same.cif", "/three/same.cif"}
    summary = analyze_records(rows, tmp_path, spec())
    assert summary["status"] == "partial"
    assert len(list(csv.DictReader((tmp_path / "results.csv").open()))) == len(rows)


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -1, True])
def test_invalid_values_become_failures(value):
    result = normalize_record(row(uptake=value))
    assert result["status"] == "failure"
    assert result["uptake_in_mol_kg"] is None


def test_mock_zero_pressure_and_close_conditions():
    rows = [row(pressure=0, uptake=0), row(pressure=0.001, uptake=100),
            row("/two/same.cif", pressure=0, uptake=2, is_mock=True)]
    ranked, excluded = rank_records(rows, GraspaAnalysis(adsorption={"temperature": 298, "pressure": 0}))
    assert ranked == [{"input_structure_file": "/one/same.cif", "absolute_uptake": 0}]
    assert excluded[0]["input_structure_file"] == "/two/same.cif"


def test_legacy_identity_uses_full_path():
    assert normalize_record({**row(), "input_structure_file": "", "cif_path": "/original/a.cif"})["input_structure_file"] == "/original/a.cif"
    with pytest.raises(ValueError, match="basename"):
        normalize_record({"cif_filename": "a.cif", "status": "success"})


def test_top_fraction_ceil_and_bounded_preview(tmp_path):
    rows = [row(f"/source/{i}.cif", uptake=i) for i in range(31)]
    summary = analyze_records(rows, tmp_path, GraspaAnalysis(
        adsorption={"temperature": 298, "pressure": 960}, top_fraction=0.2,
    ))
    assert summary["selected_structures"] == 7
    assert len(summary["preview"]) == 5
    assert len(list(csv.DictReader((tmp_path / "top_candidates.csv").open()))) == 7


def test_failed_repeat_excludes_otherwise_complete_pair():
    rows = [row(), row(pressure=320), row(pressure=320, status="failure")]
    assert not rank_records(rows, spec())[0]


def test_legacy_failure_without_conditions_excludes_complete_pair():
    rows = [row(), row(pressure=320), row(temperature_in_K=None, pressure_in_Pa=None, status="failure")]
    ranked, excluded = rank_records(rows, spec())
    assert not ranked
    assert excluded[0]["reason"] == "Record has missing conditions"


def test_large_finite_repeats_do_not_overflow(tmp_path):
    summary = analyze_records([row(uptake=1.7e308)] * 3, tmp_path, GraspaAnalysis(
        adsorption={"temperature": 298, "pressure": 960},
    ))
    assert summary["status"] == "completed"
    assert summary["preview"][0]["absolute_uptake"] == pytest.approx(1.7e308)
    assert summary["conditions"]["adsorption"] == {"temperature": 298, "pressure": 960}
