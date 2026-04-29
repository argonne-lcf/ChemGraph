"""Streamlit session-state initialisation for the ChemGraph UI."""

import logging

import streamlit as st

from ui.config import load_config

logger = logging.getLogger(__name__)


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
        # Session persistence
        "session_store": None,  # SessionStore instance (created lazily)
        "current_session_id": None,  # active session ID (str or None)
        "session_created": False,  # True once the DB row has been created
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Load config from disk on first run
    if st.session_state.config is None:
        st.session_state.config = load_config()

    # Initialise SessionStore on first run
    if st.session_state.session_store is None:
        try:
            from chemgraph.memory.store import SessionStore

            st.session_state.session_store = SessionStore()
        except Exception as exc:
            logger.warning("Failed to initialise SessionStore: %s", exc)
            # Memory will be unavailable; the UI continues without it.
