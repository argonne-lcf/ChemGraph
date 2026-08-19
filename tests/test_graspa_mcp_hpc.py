"""Tests for adsorption ensemble expansion."""

from pathlib import Path

import pytest

from chemgraph.mcp.graspa_mcp_hpc import _expand_adsorption_ensemble
from chemgraph.schemas.adsorption_schema import AdsorptionEnsembleRequest


def _runtime_config(tmp_path: Path, engine: str) -> Path:
    path = tmp_path / f"{engine}.toml"
    path.write_text(
        "[adsorption]\n"
        f'engine = "{engine}"\n'
        'executable = "/worker/graspa"\n',
        encoding="utf-8",
    )
    return path


def test_cuda_expansion_uses_unique_condition_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    config = _runtime_config(tmp_path, "graspa_cuda")
    monkeypatch.setenv("CHEMGRAPH_CONFIG", str(config))
    first = tmp_path / "a.cif"
    second = tmp_path / "b.cif"
    first.write_text("data_a", encoding="utf-8")
    second.write_text("data_b", encoding="utf-8")
    params = AdsorptionEnsembleRequest(
        input_structures=[str(first), str(second)],
        output_result_file=str(tmp_path / "raspa.log"),
        conditions=[
            {"temperature": 298, "pressure": 100000},
            {"temperature": 323, "pressure": 200000},
        ],
        components=[
            {"name": "CO2", "mole_fraction": 0.15},
            {"name": "N2", "mole_fraction": 0.85},
        ],
    )
    jobs = _expand_adsorption_ensemble(params)
    assert len(jobs) == 4
    outputs = {job["output_result_file"] for job in jobs}
    assert len(outputs) == 4
    assert all("CO2-N2" in output for output in outputs)
    assert all(job["_runtime"]["engine"] == "graspa_cuda" for job in jobs)


def test_explicit_remote_files_do_not_require_directory_probe(
    tmp_path: Path, monkeypatch
) -> None:
    config = _runtime_config(tmp_path, "graspa_cuda")
    monkeypatch.setenv("CHEMGRAPH_CONFIG", str(config))
    params = AdsorptionEnsembleRequest(
        remote_structure_files=["/remote/a.cif", "/remote/b.cif"],
        output_result_file="/remote/results/raspa.log",
        components=[{"name": "H2O"}],
    )
    jobs = _expand_adsorption_ensemble(params)
    assert [job["remote_structure_file"] for job in jobs] == [
        "/remote/a.cif",
        "/remote/b.cif",
    ]
    assert all("input_structure_file" not in job for job in jobs)


def test_sycl_mixture_fails_before_submission(tmp_path: Path, monkeypatch) -> None:
    config = _runtime_config(tmp_path, "graspa_sycl")
    monkeypatch.setenv("CHEMGRAPH_CONFIG", str(config))
    cif = tmp_path / "a.cif"
    cif.write_text("data_a", encoding="utf-8")
    params = AdsorptionEnsembleRequest(
        input_structures=[str(cif)],
        components=[
            {"name": "CO2", "mole_fraction": 0.15},
            {"name": "N2", "mole_fraction": 0.85},
        ],
    )
    with pytest.raises(ValueError, match="Use 'graspa_cuda'"):
        _expand_adsorption_ensemble(params)
