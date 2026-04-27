"""Streamlit session-state initialisation for the ChemGraph UI."""

import streamlit as st

from ui.config import load_config


def init_session_state() -> None:
    """Ensure all required session-state keys exist with sensible defaults.

    This function is idempotent -- calling it multiple times is safe.
    """
    defaults = {
        "agent": None,
        "conversation_history": [],
        "last_config": None,
        "config": None,  # loaded lazily below
        "last_run_error": None,
        "last_run_result": None,
        "last_run_query": None,
        "ui_notice": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Load config from disk on first run
    if st.session_state.config is None:
        st.session_state.config = load_config()
