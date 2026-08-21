"""Reusable Streamlit widgets for provider credentials.

Shared between the Configuration page and the first-run setup on the
main page so both render identical flows (notably the ALCF Globus
login, which needs two steps and state carried across reruns).
"""

from __future__ import annotations

import os

import streamlit as st

from ui import alcf_auth

# Session-state slots holding the NativeAppAuthClient and its sign-in URL
# between the "start login" and "complete login" reruns. Both are global
# (not per-page) so a login started on one page can be finished on another.
_PENDING_CLIENT_KEY = "_alcf_pending_login_client"
_PENDING_URL_KEY = "_alcf_pending_login_url"


def apply_api_key(env_var: str, value: str) -> bool:
    """Set a provider API key for this Streamlit process.

    Parameters
    ----------
    env_var : str
        Environment variable the model loader reads.
    value : str
        Key text from the input widget.

    Returns
    -------
    bool
        ``True`` when a non-empty key was applied.
    """
    clean = (value or "").strip()
    if not clean:
        return False
    os.environ[env_var] = clean
    return True


def clear_api_key(env_var: str) -> None:
    """Remove a provider API key from the process environment."""
    os.environ.pop(env_var, None)


def render_alcf_login(key_prefix: str) -> bool:
    """Render the two-step Globus login for ALCF inference endpoints.

    Parameters
    ----------
    key_prefix : str
        Unique widget-key prefix (the widget set appears on two pages).

    Returns
    -------
    bool
        ``True`` when a login completed during this rerun.
    """
    status = alcf_auth.token_status()
    pending = st.session_state.get(_PENDING_CLIENT_KEY)

    if status["state"] in ("env", "valid", "refreshable") and not pending:
        st.success(status["detail"])
        if st.button("Log out", key=f"{key_prefix}_alcf_logout"):
            alcf_auth.logout()
            st.rerun()
        return False

    if pending is None:
        st.caption(status["detail"])
        if st.button(
            "\U0001f510 Log in with Globus", key=f"{key_prefix}_alcf_start"
        ):
            try:
                client, url = alcf_auth.start_login()
            except Exception as exc:
                st.error(f"Could not start the Globus flow: {exc}")
                return False
            st.session_state[_PENDING_CLIENT_KEY] = client
            st.session_state[_PENDING_URL_KEY] = url
            st.rerun()
        return False

    url = st.session_state.get(_PENDING_URL_KEY, "")
    st.markdown(
        f"1. [Open the Globus sign-in page]({url}) and log in with an "
        "identity that has ALCF access.\n"
        "2. Copy the authorization code shown at the end and paste it "
        "below."
    )
    code = st.text_input(
        "Authorization code",
        key=f"{key_prefix}_alcf_code",
        type="password",
    )
    col_ok, col_cancel = st.columns(2)
    completed = False
    with col_ok:
        if st.button("Complete login", key=f"{key_prefix}_alcf_complete"):
            try:
                alcf_auth.complete_login(pending, code)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.session_state[_PENDING_CLIENT_KEY] = None
                st.session_state[_PENDING_URL_KEY] = None
                st.success("Logged in to ALCF inference endpoints.")
                completed = True
    with col_cancel:
        if st.button("Cancel", key=f"{key_prefix}_alcf_cancel"):
            st.session_state[_PENDING_CLIENT_KEY] = None
            st.session_state[_PENDING_URL_KEY] = None
            st.rerun()
    return completed
