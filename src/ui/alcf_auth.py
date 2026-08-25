"""Globus authentication for ALCF inference endpoints, usable from the UI.

Implements the same Native App OAuth flow as ALCF's official
``inference_auth_token.py`` helper (see
https://github.com/argonne-lcf/inference-endpoints) so users can log in
from the browser UI instead of running a script and pasting tokens:

- ``start_login`` returns a Globus URL; the user signs in and copies the
  authorization code back into the UI; ``complete_login`` exchanges it.
- Tokens are cached in ``~/.chemgraph/alcf_inference_tokens.json`` and
  the access token is exported as ``ALCF_ACCESS_TOKEN`` for the model
  loader (:mod:`chemgraph.models.alcf_endpoints`).
- ``ensure_access_token`` silently refreshes expired access tokens with
  the cached refresh token. Tokens created by the official helper script
  (``~/.globus/app/.../inference_app/tokens.json``) are picked up too,
  so users who already authenticated on the CLI never see a login.

Everything except the two interactive login calls is network-free and
Streamlit-free so it can be unit-tested.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Public Native App client and gateway (resource server) published by ALCF
# for the inference endpoints. Same IDs as inference_auth_token.py.
AUTH_CLIENT_ID = "58fdd3bc-e1c3-4ce5-80ea-8d6b87cfb944"
GATEWAY_CLIENT_ID = "681c10cc-f684-4540-bcd7-0b4df3bc26ef"
GATEWAY_SCOPE = f"https://auth.globus.org/scopes/{GATEWAY_CLIENT_ID}/action_all"
# ALCF requires an identity from this Globus authentication policy.
SESSION_POLICY = "83732ff2-9c42-4548-b5ce-17e498c84f6a"

TOKEN_ENV = "ALCF_ACCESS_TOKEN"

# ChemGraph-owned token cache (flat record, chmod 600).
CHEMGRAPH_TOKENS_PATH = os.path.expanduser(
    "~/.chemgraph/alcf_inference_tokens.json"
)
# Cache written by ALCF's official helper (globus_sdk.UserApp storage).
HELPER_TOKENS_PATH = os.path.expanduser(
    f"~/.globus/app/{AUTH_CLIENT_ID}/inference_app/tokens.json"
)

# Refuse tokens that expire within this margin so a request started now
# does not die mid-flight.
_EXPIRY_MARGIN_S = 60


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _record_from_helper_store(data: dict) -> Optional[dict]:
    """Extract a flat token record from the UserApp storage format."""
    nested = (
        data.get("data", {}).get("DEFAULT", {}).get(GATEWAY_CLIENT_ID, {})
    )
    if isinstance(nested, dict) and nested.get("access_token"):
        return nested
    return None


def read_token_record() -> tuple[Optional[dict], Optional[str]]:
    """Return the cached token record and its source path, if any.

    The ChemGraph cache wins over the helper-script cache because it is
    the one this module keeps fresh.

    Returns
    -------
    tuple[dict | None, str | None]
        ``(record, path)`` or ``(None, None)`` when no cache exists.
    """
    data = _load_json(CHEMGRAPH_TOKENS_PATH)
    if isinstance(data, dict) and data.get("access_token"):
        return data, CHEMGRAPH_TOKENS_PATH

    data = _load_json(HELPER_TOKENS_PATH)
    if isinstance(data, dict):
        record = _record_from_helper_store(data)
        if record:
            return record, HELPER_TOKENS_PATH
    return None, None


def _expires_at(record: dict) -> Optional[float]:
    value = record.get("expires_at_seconds")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_record_valid(record: dict, now: Optional[float] = None) -> bool:
    """Return whether a token record's access token is still usable.

    Records without an expiry are treated as expired so the refresh
    path (which learns the real expiry) runs instead.

    Parameters
    ----------
    record : dict
        Flat token record.
    now : float, optional
        Current UNIX time (defaults to ``time.time()``); test hook.

    Returns
    -------
    bool
        ``True`` when the access token is present and not near expiry.
    """
    if not record.get("access_token"):
        return False
    expires_at = _expires_at(record)
    if expires_at is None:
        return False
    if now is None:
        now = time.time()
    return expires_at - now > _EXPIRY_MARGIN_S


def save_token_record(record: dict) -> None:
    """Persist a flat token record to the ChemGraph cache (chmod 600)."""
    path = Path(CHEMGRAPH_TOKENS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    os.chmod(path, 0o600)


def _export(record: dict) -> Optional[str]:
    token = record.get("access_token")
    if token:
        os.environ[TOKEN_ENV] = token
    return token


def _refresh_record(record: dict) -> Optional[dict]:
    """Exchange a refresh token for a fresh access token (network call)."""
    refresh_token = record.get("refresh_token")
    if not refresh_token:
        return None
    try:
        import globus_sdk

        client = globus_sdk.NativeAppAuthClient(AUTH_CLIENT_ID)
        response = client.oauth2_refresh_token(refresh_token)
        fresh = response.by_resource_server.get(GATEWAY_CLIENT_ID)
    except Exception as exc:
        logger.warning("ALCF token refresh failed: %s", exc)
        return None
    if not fresh or not fresh.get("access_token"):
        return None
    fresh = dict(fresh)
    # Globus may omit the refresh token on refresh responses.
    fresh.setdefault("refresh_token", refresh_token)
    return fresh


def ensure_access_token(allow_refresh: bool = True) -> Optional[str]:
    """Return a usable ALCF access token, exporting it to the environment.

    Order: existing ``ALCF_ACCESS_TOKEN`` env var, then a still-valid
    cached record, then (optionally) a silent refresh with the cached
    refresh token.

    Parameters
    ----------
    allow_refresh : bool, optional
        Permit the network refresh call for expired records.

    Returns
    -------
    str or None
        Access token, or ``None`` when the user must log in.
    """
    env_token = os.environ.get(TOKEN_ENV)
    record, _source = read_token_record()

    # A token supplied via the environment (not one this module exported
    # from the cache) is externally managed: trust it as-is.
    if env_token and (record is None or record.get("access_token") != env_token):
        return env_token

    if record is None:
        return None
    if is_record_valid(record):
        return _export(record)
    if not allow_refresh:
        return None

    fresh = _refresh_record(record)
    if fresh is None:
        return None
    try:
        save_token_record(fresh)
    except OSError as exc:
        logger.warning("Could not cache refreshed ALCF token: %s", exc)
    return _export(fresh)


def token_status() -> dict[str, Any]:
    """Describe the current authentication state without network calls.

    Returns
    -------
    dict[str, Any]
        ``state`` is one of ``"env"`` (token provided via environment),
        ``"valid"``, ``"refreshable"``, or ``"logged_out"``; ``detail``
        is a human-readable summary.
    """
    env_token = os.environ.get(TOKEN_ENV)
    record, source = read_token_record()
    # Only report "env" for externally supplied tokens; a token this
    # module exported from the cache keeps expiry/refresh semantics.
    if env_token and (record is None or record.get("access_token") != env_token):
        return {
            "state": "env",
            "detail": f"Using token from ${TOKEN_ENV}.",
        }
    if record is None:
        return {"state": "logged_out", "detail": "Not logged in."}
    if is_record_valid(record):
        expires_at = _expires_at(record)
        remaining_h = (expires_at - time.time()) / 3600 if expires_at else 0
        return {
            "state": "valid",
            "detail": (
                f"Logged in (token valid for {remaining_h:.1f} h, "
                f"cached in {source})."
            ),
        }
    if record.get("refresh_token"):
        return {
            "state": "refreshable",
            "detail": "Access token expired; it will refresh automatically.",
        }
    return {"state": "logged_out", "detail": "Cached token expired."}


def start_login():
    """Begin the Globus Native App flow.

    Returns
    -------
    tuple
        ``(client, authorize_url)``. Keep *client* around (e.g. in
        Streamlit session state) and pass it to :func:`complete_login`
        together with the code the user pastes back.
    """
    import globus_sdk

    client = globus_sdk.NativeAppAuthClient(AUTH_CLIENT_ID)
    client.oauth2_start_flow(
        requested_scopes=GATEWAY_SCOPE,
        refresh_tokens=True,
    )
    url = client.oauth2_get_authorize_url(
        session_required_policies=[SESSION_POLICY]
    )
    return client, url


def complete_login(client, auth_code: str) -> str:
    """Exchange the pasted authorization code for tokens.

    Parameters
    ----------
    client : globus_sdk.NativeAppAuthClient
        Client returned by :func:`start_login`.
    auth_code : str
        Authorization code copied from the Globus page.

    Returns
    -------
    str
        The new access token (also cached and exported to the env).

    Raises
    ------
    RuntimeError
        When the exchange fails or returns no gateway tokens.
    """
    if not auth_code or not auth_code.strip():
        raise RuntimeError("Authorization code is empty.")
    try:
        response = client.oauth2_exchange_code_for_tokens(auth_code.strip())
    except Exception as exc:
        raise RuntimeError(f"Globus rejected the authorization code: {exc}")
    record = response.by_resource_server.get(GATEWAY_CLIENT_ID)
    if not record or not record.get("access_token"):
        raise RuntimeError(
            "Globus returned no tokens for the ALCF inference gateway. "
            "The account may not satisfy ALCF's access policy."
        )
    record = dict(record)
    try:
        save_token_record(record)
    except OSError as exc:
        # The one-time code is already spent; a cache-write failure must
        # not lose the session's token.
        logger.warning("Could not cache ALCF tokens: %s", exc)
    token = _export(record)
    return token


def logout() -> None:
    """Forget the ChemGraph-cached tokens and the exported env token.

    The official helper script's own cache is left untouched.
    """
    os.environ.pop(TOKEN_ENV, None)
    try:
        os.remove(CHEMGRAPH_TOKENS_PATH)
    except OSError:
        pass
