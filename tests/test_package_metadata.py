"""Tests for ChemGraph package metadata."""

import re
import io
import runpy
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import chemgraph
import pytest
from packaging.requirements import Requirement

_REPO_ROOT = Path(__file__).resolve().parents[1]
_EXACT_PIN = re.compile(
    r"^([A-Za-z0-9_.-]+)(?:==|=)([^;\s]+)(?:\s*;.*)?$"
)


def _normalize_package_name(name: str) -> str:
    """Return a normalized Python package name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _exact_pins(requirements: list[str]) -> dict[str, str]:
    """Return normalized names and versions for exact package pins."""
    pins = {}
    for requirement in requirements:
        match = _EXACT_PIN.fullmatch(requirement.strip())
        if match:
            pins[_normalize_package_name(match.group(1))] = match.group(2)
    return pins


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML data for tests.

    Parameters
    ----------
    path : pathlib.Path
        TOML file to read.

    Returns
    -------
    dict[str, Any]
        Parsed TOML data.
    """
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib

        return tomllib.loads(text)
    except ModuleNotFoundError:
        import toml

        return toml.loads(text)


def test_version_matches_project_metadata() -> None:
    """ChemGraph version should match the project metadata version."""
    pyproject = _REPO_ROOT / "pyproject.toml"
    project_version = _load_toml(pyproject)["project"]["version"]

    assert chemgraph.__version__ == project_version


def test_pyproject_version_fallback_matches_project_metadata() -> None:
    """Source-checkout fallback should read the version from pyproject.toml."""
    pyproject = _REPO_ROOT / "pyproject.toml"
    project_version = _load_toml(pyproject)["project"]["version"]

    assert chemgraph._version_from_pyproject() == project_version


def test_conda_environment_covers_exact_project_pins() -> None:
    """Conda installs should retain every exact core dependency pin."""
    project = _load_toml(_REPO_ROOT / "pyproject.toml")["project"]
    project_pins = _exact_pins(project["dependencies"])

    environment_text = (_REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    environment_specs = re.findall(
        r"^\s*-\s+(.+?)\s*$", environment_text, flags=re.MULTILINE
    )
    environment_pins = _exact_pins(environment_specs)

    mismatches = {
        name: {"project": version, "environment": environment_pins.get(name)}
        for name, version in project_pins.items()
        if environment_pins.get(name) != version
    }
    assert not mismatches

    assert "-r requirements/mace-polar.txt" in environment_specs


def test_calculator_pin_matches_all_installation_surfaces() -> None:
    """Conda and container installs should use the calculator-extra TBLite pin."""
    metadata = _load_toml(_REPO_ROOT / "pyproject.toml")["project"]
    calculator_pins = _exact_pins(metadata["optional-dependencies"]["calculators"])
    tblite_version = calculator_pins["tblite"]

    environment_text = (_REPO_ROOT / "environment.yml").read_text(encoding="utf-8")
    environment_specs = re.findall(
        r"^\s*-\s+(.+?)\s*$", environment_text, flags=re.MULTILINE
    )
    assert _exact_pins(environment_specs)["tblite"] == tblite_version

    for filename in ("Dockerfile", "Dockerfile.arm"):
        dockerfile = (_REPO_ROOT / filename).read_text(encoding="utf-8")
        assert f'"tblite=={tblite_version}"' in dockerfile
        assert "python -m pip check" in dockerfile
        assert "-r requirements/mace-polar.txt" in dockerfile


def test_published_dependencies_have_no_direct_urls():
    project = _load_toml(_REPO_ROOT / "pyproject.toml")["project"]
    groups = [project["dependencies"], *project["optional-dependencies"].values()]
    assert all(Requirement(dep).url is None for group in groups for dep in group)


def test_supplemental_dependency_pins():
    expected = {
        "mace-polar.txt": [
            "graph-longrange @ git+https://github.com/WillBaldwin0/graph_electrostatics.git@v0.4.0",
        ],
        "ocsr-models.txt": [
            "glyph @ git+https://github.com/EdisonScientific/glyph@0bf782f863d26b041ace157668928ef07c38b972",
            "MolNexTR @ git+https://github.com/reowszer/MolNexTR@f450b9661557b1f91ae36f59dd1fadfbcb3a0967",
            "MolScribe @ git+https://github.com/reowszer/MolScribe@b03b30fbac9a78434116e626ebef6c7b7bdcdb6e",
        ],
    }
    for filename, pins in expected.items():
        lines = (_REPO_ROOT / "requirements" / filename).read_text().splitlines()
        assert [line for line in lines if line and not line.startswith("#")] == pins


@pytest.mark.parametrize("kind", ["whl", "tar.gz"])
@pytest.mark.parametrize("requirement", [
    "numpy>=2", "engine @ https://example.org/engine.whl",
    'engine @ git+https://example.org/engine.git@v1 ; extra == "optional"',
])
def test_built_metadata_rejects_urls_including_extras(tmp_path, kind, requirement):
    check = runpy.run_path(str(_REPO_ROOT / "scripts/check_distribution_metadata.py"))[
        "check_distribution"
    ]
    path = tmp_path / f"chemgraph-0.6.0.{kind}"
    metadata = f"Metadata-Version: 2.4\nRequires-Dist: {requirement}\n".encode()
    if kind == "whl":
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("chemgraph-0.6.0.dist-info/METADATA", metadata)
    else:
        with tarfile.open(path, "w:gz") as archive:
            for name, data in {
                "PKG-INFO": metadata, "requirements/mace-polar.txt": b"",
                "requirements/ocsr-models.txt": b"",
            }.items():
                member = tarfile.TarInfo(f"chemgraph-0.6.0/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
    if Requirement(requirement).url:
        with pytest.raises(ValueError, match="direct-URL dependency"):
            check(path)
    else:
        check(path)
