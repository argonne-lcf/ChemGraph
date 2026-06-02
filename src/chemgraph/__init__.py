"""ChemGraph package metadata."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

_DISTRIBUTION_NAMES = ("chemgraphagent", "chemgraph")


def _load_toml(path: Path) -> dict[str, Any]:
    """Load TOML data from a file path.

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


def _version_from_distribution() -> str | None:
    """Return the installed package version when distribution metadata exists.

    Returns
    -------
    str or None
        Installed version string, or ``None`` when ChemGraph is running from a
        source checkout without package metadata.
    """
    for dist_name in _DISTRIBUTION_NAMES:
        try:
            return version(dist_name)
        except PackageNotFoundError:
            continue
    return None


def _version_from_pyproject() -> str | None:
    """Return the project version from a source checkout's ``pyproject.toml``.

    Returns
    -------
    str or None
        Project version string, or ``None`` when the file cannot be found or
        parsed.
    """
    start = Path(__file__).resolve()
    for directory in start.parents:
        pyproject = directory / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            project = _load_toml(pyproject).get("project", {})
        except Exception:
            return None
        project_version = project.get("version")
        return str(project_version) if project_version else None
    return None


__version__ = _version_from_pyproject() or _version_from_distribution() or "unknown"
