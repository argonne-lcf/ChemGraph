"""Packaged Agent Skills for ChemGraph users and HPC workflows."""

from pathlib import Path


def skill_directory() -> str:
    """Return the installed skill root for filesystem-backed integrations."""
    return str(Path(__file__).resolve().parent)
