"""Branding assets for the ChemGraph Streamlit UI."""

from pathlib import Path


_UI_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ASSET_DIR = _UI_DIR / "assets"
_LOGO_DIR = _PROJECT_ROOT / "logo"

LOGO_IMAGES = (
    _ASSET_DIR / "chemgraph-logo.png",
    _LOGO_DIR / "chemgraph-color-dark__rgb-lores.png",
)
ICON_IMAGES = (
    _ASSET_DIR / "chemgraph-icon.png",
    _LOGO_DIR / "chemgraph-icon-color-dark__rgb-lores.png",
)


def first_existing_asset(paths: tuple[Path, ...]) -> str | None:
    """Return the first available Streamlit-compatible asset path."""
    for path in paths:
        if path.exists():
            return str(path)
    return None
