"""Local model endpoint health-check utilities."""

from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

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


def _build_local_models_probe_url(base_url: str) -> Optional[str]:
    """Validate and canonicalize a local endpoint probe URL.

    Returns a safe canonical URL ending in ``/models`` when valid,
    otherwise ``None``.
    """
    parsed = urlparse((base_url or "").strip())
    if parsed.scheme not in {"http", "https"}:
        return None
    if not _is_local_address(parsed.hostname or ""):
        return None
    # Disallow userinfo, query, and fragment in probe target.
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        return None

    base_path = parsed.path.rstrip("/")
    safe_path = f"{base_path}/models" if base_path else "/models"
    netloc = parsed.hostname or ""
    if parsed.port:
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
    if not base_url:
        return {"ok": True, "message": "No base URL configured."}

    probe = _build_local_models_probe_url(base_url)
    if not probe:
        return {"ok": False, "message": "Invalid local endpoint URL."}

    req = Request(probe, method="GET")

    try:
        with urlopen(req, timeout=2) as response:
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
