 
"""ChemGraph package metadata."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("chemgraphagent")
except PackageNotFoundError:
    # Local source tree without installed package metadata.
    __version__ = "unknown"
