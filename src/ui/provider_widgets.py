"""Reusable Streamlit widgets for provider credentials.

Shared between the Configuration page and the first-run setup on the
main page so both render identical flows (notably the ALCF Globus
login, which needs two steps and state carried across reruns).
"""

from __future__ import annotations

import os

import streamlit as st

from ui import alcf_auth
from ui import codex_auth

# Session-state slots holding the NativeAppAuthClient and its sign-in URL
# between the "start login" and "complete login" reruns. Both are global
# (not per-page) so a login started on one page can be finished on another.
_PENDING_CLIENT_KEY = "_alcf_pending_login_client"
_PENDING_URL_KEY = "_alcf_pending_login_url"

# Session-state slot holding the running Codex device-code login
# handle between reruns (global for the same reason as the ALCF slots).
_CODEX_LOGIN_KEY = "_codex_pending_device_login"


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


def render_codex_login(key_prefix: str) -> bool:
    """Render Codex (ChatGPT subscription) login status and device-code login.

    Mirrors the CLI: the model loader accepts only a ChatGPT-managed Codex
    login.  The widget shows that status and, on request, starts a
    device-code login through the pinned Codex SDK, displaying the
    verification URL and one-time code so the login can be completed from
    any browser (the Streamlit server may be remote/headless).

    Parameters
    ----------
    key_prefix : str
        Unique widget-key prefix (the widget set appears on two pages).

    Returns
    -------
    bool
        ``True`` when a login completed during this rerun.
    """
    pending = st.session_state.get(_CODEX_LOGIN_KEY)

    # ----- A device-code login is in progress -----
    if pending is not None:
        if pending.finished:
            st.session_state[_CODEX_LOGIN_KEY] = None
            codex_auth.invalidate_status_cache()
            if pending.succeeded:
                st.success("Signed in to Codex with ChatGPT.")
                return True
            if not pending.cancelled:
                st.error(f"Codex login did not complete: {pending.output}")
            return False

        st.info(
            "Finish signing in with ChatGPT, then come back here.",
            icon="\U0001f510",
        )
        st.markdown(
            f"1. Open [{pending.url}]({pending.url}) and sign in to the ChatGPT "
            "account that has Codex access.\n"
            "2. Enter this one-time code when asked:"
        )
        st.code(pending.code, language="text")
        col_check, col_cancel = st.columns(2)
        with col_check:
            if st.button("I have signed in", key=f"{key_prefix}_codex_check"):
                # Give the SDK a moment to deliver the completion notice.
                pending.wait(timeout=5.0)
                st.rerun()
        with col_cancel:
            if st.button("Cancel", key=f"{key_prefix}_codex_cancel"):
                pending.cancel()
                st.session_state[_CODEX_LOGIN_KEY] = None
                codex_auth.invalidate_status_cache()
                st.rerun()
        return False

    # ----- No login in progress: show status and actions -----
    status = codex_auth.account_status()
    if status.ready:
        st.success(status.detail)
        if st.button("Log out", key=f"{key_prefix}_codex_logout"):
            ok, output = codex_auth.logout()
            if not ok:
                st.error(f"Codex logout failed: {output}")
            else:
                st.rerun()
        return False

    if status.state == codex_auth.STATE_NO_SDK:
        st.warning(status.detail)
        st.caption(codex_auth.INSTALL_HINT)
        return False
    if status.state == codex_auth.STATE_ERROR:
        st.error(status.detail)
    else:
        st.caption(status.detail)

    col_login, col_refresh = st.columns(2)
    with col_login:
        label = (
            "Log out and sign in with ChatGPT"
            if status.state == codex_auth.STATE_API_KEY
            else "\U0001f510 Sign in with ChatGPT"
        )
        if st.button(label, key=f"{key_prefix}_codex_start"):
            if status.state == codex_auth.STATE_API_KEY:
                ok, output = codex_auth.logout()
                if not ok:
                    st.error(f"Codex logout failed: {output}")
                    return False
            with st.spinner("Requesting a sign-in code from Codex..."):
                try:
                    st.session_state[_CODEX_LOGIN_KEY] = codex_auth.start_device_login()
                except RuntimeError as exc:
                    st.error(str(exc))
                    return False
            st.rerun()
    with col_refresh:
        if st.button("Re-check login", key=f"{key_prefix}_codex_refresh"):
            codex_auth.invalidate_status_cache()
            st.rerun()
    st.caption(
        "Alternatively run `codex login` in a terminal on the machine hosting "
        "this UI, then click **Re-check login**."
    )
    return False
