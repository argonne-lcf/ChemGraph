"""Local model endpoint health-check utilities."""

from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import ParseResult, urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

import streamlit as st


def _is_local_address(hostname: str) -> bool:
    """Return whether a hostname points to the local machine.

    Parameters
    ----------
    hostname : str
        Hostname parsed from a URL.

    Returns
    -------
    bool
        ``True`` for localhost-style addresses.
    """
    host = (hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


class _NoRedirectHandler(HTTPRedirectHandler):
    """Keep local probes from following redirects to other endpoints."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _build_local_models_probe_url(parsed: ParseResult) -> Optional[str]:
    """Validate and canonicalize a local endpoint probe URL.

    Returns a safe canonical URL ending in ``/models`` when valid,
    otherwise ``None``.
    """
    if parsed.scheme not in {"http", "https"}:
        return None
    if not _is_local_address(parsed.hostname or ""):
        return None
    # Disallow userinfo, query, and fragment in probe target.
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return None

    base_path = parsed.path.rstrip("/")
    safe_path = f"{base_path}/models" if base_path else "/models"
    netloc = parsed.hostname or ""
    if ":" in netloc:
        netloc = f"[{netloc}]"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse((parsed.scheme, netloc, safe_path, "", "", ""))


@st.cache_data(ttl=10)
def check_local_model_endpoint(base_url: Optional[str]) -> Dict[str, Any]:
    """Quick reachability check for local OpenAI-compatible endpoints.

    Parameters
    ----------
    base_url : str, optional
        Base URL to probe.

    Returns
    -------
    dict[str, Any]
        Status dictionary with ``ok`` and ``message`` keys.
    """
    base_url = (base_url or "").strip()
    if not base_url:
        return {"ok": True, "message": "No base URL configured."}

    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        # Port validation is lazy in urllib, including for remote URLs.
        _ = parsed.port
    except ValueError:
        return {"ok": False, "message": "Invalid endpoint URL."}
    if parsed.scheme not in {"http", "https"} or not hostname:
        return {"ok": False, "message": "Invalid endpoint URL."}
    if not _is_local_address(hostname):
        return {"ok": True, "message": "Skipping non-local endpoint probe."}

    probe = _build_local_models_probe_url(parsed)
    if not probe:
        return {"ok": False, "message": "Invalid local endpoint URL."}

    try:
        req = Request(probe, method="GET")
        with build_opener(_NoRedirectHandler()).open(req, timeout=2) as response:
            code = getattr(response, "status", 200)
            return {"ok": True, "message": f"Reachable (HTTP {code})."}
    except HTTPError as e:
        # HTTP error still means service/socket is reachable.
        return {"ok": True, "message": f"Reachable (HTTP {e.code})."}
    except URLError as e:
        reason = getattr(e, "reason", e)
        return {"ok": False, "message": f"Unreachable: {reason}"}
    except Exception as e:
        return {"ok": False, "message": f"Unreachable: {e}"}
