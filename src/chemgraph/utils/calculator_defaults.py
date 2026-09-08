"""Lightweight calculator defaults shared by schemas and the UI."""

import importlib.util


def mace_polar_available() -> bool:
    """Detect the Polar add-on without importing engines or loading weights."""
    try:
        return importlib.util.find_spec("graph_longrange") is not None
    except (ImportError, ValueError):
        return False


def get_default_mace_calculator_type() -> str:
    """Prefer Polar when its add-on is installed, otherwise use MACE-MP."""
    return "mace_polar" if mace_polar_available() else "mace_mp"
