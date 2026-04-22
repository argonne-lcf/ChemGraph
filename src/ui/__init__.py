"""ChemGraph UI Package.

This package contains the Streamlit web application for ChemGraph.
The CLI has been moved to ``chemgraph.cli``.
"""

try:
    from chemgraph import __version__
except Exception:
    __version__ = "unknown"
