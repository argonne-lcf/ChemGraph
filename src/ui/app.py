"""ChemGraph Streamlit application entry point.

Run with:  ``streamlit run src/ui/app.py``

This thin module handles page configuration (which **must** be the first
Streamlit call), sidebar navigation, and page dispatch.  All page content
lives in :mod:`ui.pages`.
"""

import sys
from pathlib import Path

# Ensure the parent of ui/ (i.e. src/) is on sys.path so that
# "from ui.xxx import ..." works when run as a standalone script.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st

from chemgraph import __version__ as chemgraph_version

from ui.system_info import render_sidebar_host_and_build_info
from ui.visualization import warn_stmol_unavailable

# ---------------------------------------------------------------------------
# Page configuration -- MUST be the first Streamlit call
# ---------------------------------------------------------------------------
app_version = (
    chemgraph_version
    if isinstance(chemgraph_version, str) and chemgraph_version != "unknown"
    else "dev"
)

st.set_page_config(
    page_title="ChemGraph",
    page_icon="\U0001f9ea",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One-time stmol availability warning
warn_stmol_unavailable()

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("\U0001f9ea ChemGraph")
page = st.sidebar.radio(
    "Navigate",
    ["\U0001f3e0 Main Interface", "\u2699\ufe0f Configuration", "\U0001f4d6 About ChemGraph"],
    index=0,
    key="page_navigation",
)
render_sidebar_host_and_build_info()

# ---------------------------------------------------------------------------
# Page dispatch
# ---------------------------------------------------------------------------
if page == "\U0001f4d6 About ChemGraph":
    from ui._pages import about

    about.render()
    st.stop()

elif page == "\u2699\ufe0f Configuration":
    from ui._pages import configuration

    configuration.render()
    st.stop()

else:
    from ui._pages import main_interface

    main_interface.render()
