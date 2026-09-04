"""Tests for ChemGraph package metadata."""

import re
from pathlib import Path
from typing import Any

import chemgraph

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

    direct_urls = [dep for dep in project["dependencies"] if " @ " in dep]
    assert all(dependency in environment_specs for dependency in direct_urls)


def test_tblite_pin_matches_all_installation_surfaces() -> None:
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
