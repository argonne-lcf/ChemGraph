"""ChemGraph Streamlit application entry point.

Run with:  ``streamlit run src/ui/app.py``  (or ``chemgraph ui``)

This thin module handles page configuration (which **must** be the first
Streamlit call) and navigation via ``st.navigation``, which gives each
page its own URL and a native sidebar entry.  All page content lives in
:mod:`ui._pages`.
"""

import sys
from pathlib import Path

# Ensure the parent of ui/ (i.e. src/) is on sys.path so that
# "from ui.xxx import ..." works when run as a standalone script.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import streamlit as st  # noqa: E402

from chemgraph import __version__ as chemgraph_version  # noqa: E402

from ui.branding import ICON_IMAGES, LOGO_IMAGES, first_existing_asset  # noqa: E402
from ui.system_info import render_sidebar_host_and_build_info  # noqa: E402
from ui.visualization import warn_viewer_unavailable  # noqa: E402

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
    page_icon=first_existing_asset(ICON_IMAGES) or "\U0001f9ea",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One-time 3D viewer availability warning
warn_viewer_unavailable()


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def _chat_page() -> None:
    """Render the chat page."""
    from ui._pages import main_interface

    main_interface.render()


def _configuration_page() -> None:
    """Render the configuration page."""
    from ui._pages import configuration

    configuration.render()


def _about_page() -> None:
    """Render the about page."""
    from ui._pages import about

    about.render()


navigation = st.navigation(
    [
        st.Page(_chat_page, title="Chat", icon="\U0001f9ea", default=True),
        st.Page(_configuration_page, title="Configuration", icon="⚙️"),
        st.Page(_about_page, title="About", icon="\U0001f4d6"),
    ]
)

# ---------------------------------------------------------------------------
# Sidebar chrome shared by all pages
# ---------------------------------------------------------------------------
logo_image = first_existing_asset(LOGO_IMAGES)
icon_image = first_existing_asset(ICON_IMAGES)
if logo_image:
    st.logo(logo_image, icon_image=icon_image)
else:
    st.sidebar.title("\U0001f9ea ChemGraph")

render_sidebar_host_and_build_info()

navigation.run()
