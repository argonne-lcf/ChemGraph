 
"""ChemGraph package metadata."""

from importlib.metadata import PackageNotFoundError, packages_distributions, version

try:
    dist_names = packages_distributions().get("chemgraph", [])
    __version__ = version(dist_names[0]) if dist_names else "unknown"
except (PackageNotFoundError, IndexError):
    # Local source tree without installed package metadata.
    __version__ = "unknown"
