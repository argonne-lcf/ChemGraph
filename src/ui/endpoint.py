"""Local model endpoint health-check utilities."""

from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import streamlit as st


def _is_local_address(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


@st.cache_data(ttl=10)
def check_local_model_endpoint(base_url: Optional[str]) -> Dict[str, Any]:
    """Quick reachability check for local OpenAI-compatible endpoints."""
    if not base_url:
        return {"ok": True, "message": "No base URL configured."}

    parsed = urlparse(base_url)
    if not _is_local_address(parsed.hostname or ""):
        return {"ok": True, "message": "Skipping non-local endpoint probe."}

    probe = base_url.rstrip("/") + "/models"
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
