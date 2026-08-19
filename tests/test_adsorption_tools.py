"""Tests for engine-neutral adsorption execution."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest
from pydantic import ValidationError

from chemgraph.schemas.adsorption_schema import AdsorptionRequest
from chemgraph.tools.adsorption_config import load_adsorption_runtime
from chemgraph.tools.adsorption_core import run_adsorption_core
from chemgraph.tools.adsorption_drivers import get_adsorption_driver


@pytest.fixture
def charged_cif(tmp_path: Path) -> Path:
    path = tmp_path / "framework.cif"
    path.write_text(
        """data_framework
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_charge
C1 C 0 0 0 0.0
""",
        encoding="utf-8",
    )
    return path


def _executable(tmp_path: Path, output: str, exit_code: int = 0) -> Path:
    path = tmp_path / "fake-graspa"
    path.write_text(
        "#!/bin/sh\n"
        f"printf '%s' {shlex.quote(output)}\n"
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_mixture_fractions_are_required_and_normalized() -> None:
    with pytest.raises(ValidationError, match="requires mole_fraction"):
        AdsorptionRequest(
            input_structure_file="framework.cif",
            components=[{"name": "CO2"}, {"name": "N2"}],
        )
    with pytest.raises(ValidationError, match="sum to 1.0"):
        AdsorptionRequest(
            input_structure_file="framework.cif",
            components=[
                {"name": "CO2", "mole_fraction": 0.2},
                {"name": "N2", "mole_fraction": 0.7},
            ],
        )


def test_engine_options_cannot_override_managed_fields() -> None:
    with pytest.raises(ValidationError, match="managed by ChemGraph"):
        AdsorptionRequest(
            input_structure_file="framework.cif",
            components=[{"name": "CO2"}],
            engine_options={"Temperature": 500},
        )
    with pytest.raises(ValidationError, match="cannot contain newlines"):
        AdsorptionRequest(
            input_structure_file="framework.cif",
            components=[
                {"name": "CO2", "engine_options": {"WidomProbability": "1\nX"}}
            ],
        )


def test_sycl_capability_rejects_mixture(charged_cif: Path) -> None:
    request = AdsorptionRequest(
        input_structure_file=str(charged_cif),
        components=[
            {"name": "CO2", "mole_fraction": 0.15},
            {"name": "N2", "mole_fraction": 0.85},
        ],
    )
    with pytest.raises(ValueError, match="Use 'graspa_cuda'"):
        get_adsorption_driver("graspa_sycl").validate(request, charged_cif)


def test_cuda_mixture_run_and_selectivity(
    charged_cif: Path, tmp_path: Path
) -> None:
    output = """Component 0 (CO2)
Loading absolute [mol/kg framework]
Overall: Average: 2.0, Error: 0.2
Loading excess [mol/kg framework]
Overall: Average: 1.8, Error: 0.3
Component 1 (N2)
Loading absolute [mol/kg framework]
Overall: Average: 1.0, Error: 0.1
Loading excess [mol/kg framework]
Overall: Average: 0.9, Error: 0.2
"""
    executable = _executable(tmp_path, output)
    request = AdsorptionRequest(
        input_structure_file=str(charged_cif),
        output_result_file=str(tmp_path / "results" / "raspa.log"),
        components=[
            {"name": "CO2", "mole_fraction": 0.2},
            {"name": "N2", "mole_fraction": 0.8},
        ],
    )
    result = run_adsorption_core(
        request,
        runtime={
            "engine": "graspa_cuda",
            "executable": str(executable),
            "environment": {"OMP_NUM_THREADS": "1"},
        },
    )

    assert result["status"] == "success"
    assert [component["uptake"] for component in result["components"]] == [2.0, 1.0]
    assert result["selectivities"][0]["value"] == pytest.approx(8.0)
    workdir = Path(result["working_directory"])
    rendered = (workdir / "simulation.input").read_text(encoding="utf-8")
    assert "Component 0 MoleculeName              CO2" in rendered
    assert "Component 1 MoleculeName              N2" in rendered
    assert rendered.count("IdentityChangeProbability 1.0") == 2
    assert Path(result["stdout_path"]).is_file()
    assert Path(result["stderr_path"]).is_file()


def test_cuda_h2o_uses_tip4p_alias(charged_cif: Path, tmp_path: Path) -> None:
    driver = get_adsorption_driver("graspa_cuda")
    request = AdsorptionRequest(
        input_structure_file=str(charged_cif), components=[{"name": "H2O"}]
    )
    workdir = tmp_path / "stage"
    workdir.mkdir()
    driver.validate(request, charged_cif)
    driver.stage(request, charged_cif, workdir)
    rendered = (workdir / "simulation.input").read_text(encoding="utf-8")
    assert "MoleculeName              TIP4P" in rendered
    assert (workdir / "TIP4P.def").is_file()


def test_nonzero_return_code_preserves_artifacts(
    charged_cif: Path, tmp_path: Path
) -> None:
    executable = _executable(tmp_path, "failed", exit_code=7)
    result = run_adsorption_core(
        AdsorptionRequest(
            input_structure_file=str(charged_cif),
            output_result_file=str(tmp_path / "raspa.log"),
            components=[{"name": "CO2"}],
        ),
        runtime={"engine": "graspa_cuda", "executable": str(executable)},
    )
    assert result["status"] == "failure"
    assert result["return_code"] == 7
    assert "code 7" in result["message"]
    assert Path(result["stdout_path"]).read_text(encoding="utf-8") == "failed"


def test_missing_executable_is_reported_before_launch(
    charged_cif: Path, tmp_path: Path
) -> None:
    result = run_adsorption_core(
        AdsorptionRequest(
            input_structure_file=str(charged_cif),
            output_result_file=str(tmp_path / "raspa.log"),
            components=[{"name": "CO2"}],
        ),
        runtime={
            "engine": "graspa_cuda",
            "executable": str(tmp_path / "missing-graspa"),
        },
    )
    assert result["status"] == "failure"
    assert "does not exist" in result["message"]
    assert Path(result["stdout_path"]).is_file()
    assert Path(result["stderr_path"]).is_file()


def test_timeout_returns_failure_with_artifacts(
    charged_cif: Path, tmp_path: Path
) -> None:
    executable = tmp_path / "slow-graspa"
    executable.write_text("#!/bin/sh\nsleep 1\n", encoding="utf-8")
    executable.chmod(0o755)
    result = run_adsorption_core(
        AdsorptionRequest(
            input_structure_file=str(charged_cif),
            output_result_file=str(tmp_path / "raspa.log"),
            components=[{"name": "CO2"}],
        ),
        runtime={
            "engine": "graspa_cuda",
            "executable": str(executable),
            "timeout_seconds": 0.01,
        },
    )
    assert result["status"] == "failure"
    assert "exceeded" in result["message"]
    assert Path(result["working_directory"]).is_dir()


def test_sycl_n2_assets_and_parser(charged_cif: Path, tmp_path: Path) -> None:
    output = """Number of molecules
Overall: Average: 1.0, Error: 0.1
"""
    executable = _executable(tmp_path, output)
    result = run_adsorption_core(
        AdsorptionRequest(
            input_structure_file=str(charged_cif),
            output_result_file=str(tmp_path / "raspa.log"),
            components=[{"name": "N2"}],
        ),
        runtime={"engine": "graspa_sycl", "executable": str(executable)},
    )
    assert result["status"] == "success"
    assert result["components"][0]["name"] == "N2"
    assert result["components"][0]["uptake"] > 0
    workdir = Path(result["working_directory"])
    assert "N_n2" in (workdir / "pseudo_atoms.def").read_text(encoding="utf-8")
    assert "N_com" in (workdir / "force_field_mixing_rules.def").read_text(
        encoding="utf-8"
    )


def test_bundled_ewald_profile_requires_charges(tmp_path: Path) -> None:
    cif = tmp_path / "uncharged.cif"
    cif.write_text("data_uncharged", encoding="utf-8")
    request = AdsorptionRequest(
        input_structure_file=str(cif), components=[{"name": "CO2"}]
    )
    with pytest.raises(ValueError, match="_atom_site_charge"):
        get_adsorption_driver("graspa_cuda").validate(request, cif)


def test_runtime_config_and_legacy_mapping(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.toml"
    canonical.write_text(
        """[adsorption]
engine = "graspa_cuda"
executable = "/worker/nvc_main.x"
timeout_seconds = 90
[adsorption.environment]
OMP_NUM_THREADS = "1"
""",
        encoding="utf-8",
    )
    runtime = load_adsorption_runtime(str(canonical))
    assert runtime.engine == "graspa_cuda"
    assert runtime.environment == {"OMP_NUM_THREADS": "1"}

    legacy = tmp_path / "legacy.toml"
    legacy.write_text(
        """[graspa]
runtime = "sycl"
executable = "/worker/sycl.out"
""",
        encoding="utf-8",
    )
    with pytest.deprecated_call():
        assert load_adsorption_runtime(str(legacy)).engine == "graspa_sycl"


def test_runtime_environment_reaches_executable(
    charged_cif: Path, tmp_path: Path
) -> None:
    executable = tmp_path / "env-graspa"
    executable.write_text(
        "#!/bin/sh\n"
        "test \"$CHEMGRAPH_TEST_RUNTIME\" = expected || exit 9\n"
        "printf '%s' 'Component 0 (CO2)\n"
        "Loading absolute [mol/kg framework]\n"
        "Overall: Average: 1.0, Error: 0.1\n'\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    result = run_adsorption_core(
        AdsorptionRequest(
            input_structure_file=str(charged_cif),
            output_result_file=str(tmp_path / "raspa.log"),
            components=[{"name": "CO2"}],
        ),
        runtime={
            "engine": "graspa_cuda",
            "executable": str(executable),
            "environment": {"CHEMGRAPH_TEST_RUNTIME": "expected"},
        },
    )
    assert result["status"] == "success"
    assert os.path.isfile(result["stdout_path"])
