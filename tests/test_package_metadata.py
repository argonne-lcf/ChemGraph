"""Tests for ChemGraph package metadata."""

from pathlib import Path
from typing import Any

import chemgraph


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
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project_version = _load_toml(pyproject)["project"]["version"]

    assert chemgraph.__version__ == project_version


def test_pyproject_version_fallback_matches_project_metadata() -> None:
    """Source-checkout fallback should read the version from pyproject.toml."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    project_version = _load_toml(pyproject)["project"]["version"]

    assert chemgraph._version_from_pyproject() == project_version


def test_data_dependencies_use_bounded_compatible_ranges() -> None:
    """Published data dependencies should allow compatible 2.x releases."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    dependencies = set(_load_toml(pyproject)["project"]["dependencies"])

    assert {
        "numpy>=2.2.6,<3",
        "pandas>=2.2.3,<3",
        "numexpr>=2.11.0,<3",
    }.issubset(dependencies)
