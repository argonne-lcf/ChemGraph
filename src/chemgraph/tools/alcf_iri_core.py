"""Pure-Python implementation of the ALCF IRI Facility API integration.

No LangChain. No ``@tool`` decorators. Any Python caller (agent tool,
notebook, test, MCP server) can import :func:`dispatch` and drive the
IRI API directly. See :mod:`chemgraph.tools.alcf_iri_tools` for the
LangChain wrappers layered on top.

Full endpoint list: https://api.alcf.anl.gov/openapi.json (43 endpoints).
This module exposes them as seven category action-tables:

  facility | status | account | compute | filesystem | task | auth

plus a single dispatcher, :func:`dispatch`, that handles three action
kinds per category:

  action="list_actions"  -> return names + one-line descriptions of
                            every action this category supports
  action="describe"      -> return the JSON schema for one specific
                            action (params it takes, what it returns)
  action=<name>          -> actually invoke the endpoint

Auth: reads $ALCF_API_TOKEN for endpoints that need it. First-time
setup uses the ``alcf_auth`` in-chat re-auth tool -- ask the agent
to run action='start_reauth', visit the returned Globus URL, paste
the code back. After that the tool silently refreshes access tokens
for the next 30 days (Level 1, ``_try_refresh_token``); when the
refresh token itself expires the ``alcf_auth`` flow runs again.

Both flows are implemented in-process via globus_sdk (public Native
App client 8b84fc2d-...); no external CLI helper required. The
on-disk token cache lives at ``~/.globus/app/8b84fc2d-.../
alcf_facility_api_app/tokens.json``.

Deliberately read-only for now. Write endpoints (submit_job, cancel_job,
chmod, rm, mv, upload) are listed under actions as UNSAFE and refuse to
run without $ALCF_IRI_ALLOW_UNSAFE=1. HITL comes later.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

BASE_URL = "https://api.alcf.anl.gov/api/v1"
TIMEOUT_S = 20.0
UNSAFE_ENV = "ALCF_IRI_ALLOW_UNSAFE"

# Globus OAuth Native App constants. Public app client 8b84fc2d-... is
# the same ID ALCF publishes for user auth against their Facility API
# resource server 6be511f6-... . Used by both the silent refresh path
# (_try_refresh_token) and the in-chat re-auth flow (_start_reauth /
# _complete_reauth) so ChemGraph has zero runtime dependency on any
# external CLI helper.
_GLOBUS_AUTH_CLIENT_ID = "8b84fc2d-49e9-49ea-b54d-b3a29a70cf31"
_GLOBUS_SCOPE_CLIENT_ID = "6be511f6-a071-471f-9bc0-02a0d0836723"
_GLOBUS_SCOPE = (
    f"https://auth.globus.org/scopes/{_GLOBUS_SCOPE_CLIENT_ID}/filesystem"
)
_TOKENS_PATH = os.path.expanduser(
    f"~/.globus/app/{_GLOBUS_AUTH_CLIENT_ID}/alcf_facility_api_app/tokens.json"
)

# Module-global holding a pending OAuth flow between start_reauth and
# complete_reauth. Single-slot: only one re-auth can be in flight at a
# time per Streamlit process. Cleared on completion or on next start.
# ponytail: global state is fine here -- Streamlit is single-process,
# each user session runs its own Python interpreter, and re-auth is
# rare (once per 30 days when the refresh token expires).
_PENDING_AUTH_CLIENT = None

# Cache resource name -> uuid so agents can say "aurora" and we resolve.
# Populated on first live lookup against /status/resources. Never
# preloaded -- dynamic is the only source of truth. If ALCF adds a
# machine or rotates a UUID, the tool picks it up automatically on
# the next lookup.
_RESOURCE_CACHE: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Auth + HTTP helpers
# ---------------------------------------------------------------------------


def _headers(needs_auth: bool = True) -> dict[str, str]:
    if not needs_auth:
        return {}
    token = os.environ.get("ALCF_API_TOKEN")
    if not token and _try_refresh_token():
        token = os.environ.get("ALCF_API_TOKEN")
    if not token:
        raise RuntimeError(
            "This action needs $ALCF_API_TOKEN. Get one from "
            "https://github.com/argonne-lcf/inference-endpoints (same "
            "token works for both Inference Service and Facility API)."
        )
    return {"Authorization": f"Bearer {token}"}


def _resource_id(name: str) -> str:
    """Accept 'aurora' or the uuid itself. Case-insensitive on names.

    Resolves via the live /status/resources endpoint. No hardcoded
    fallback -- if ALCF's status endpoint is unreachable, resolution
    fails and the caller sees the HTTPError. That's the honest signal
    (the whole IRI API is probably degraded anyway; better to fail
    fast than serve stale UUIDs).
    """
    if len(name) == 36 and name.count("-") == 4:  # already a uuid
        return name
    key = name.strip().lower()
    if key in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[key]
    with httpx.Client(timeout=TIMEOUT_S) as c:
        r = c.get(f"{BASE_URL}/status/resources")
        r.raise_for_status()
        for res in r.json():
            _RESOURCE_CACHE[res["name"].strip().lower()] = res["id"]
    if key in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[key]
    raise ValueError(
        f"Unknown ALCF resource {name!r}; live /status/resources "
        f"returned: {sorted(_RESOURCE_CACHE)}"
    )


def _check_unsafe(action: str) -> None:
    if not os.environ.get(UNSAFE_ENV):
        raise RuntimeError(
            f"Action {action!r} modifies HPC state. Refusing to run "
            f"without ${UNSAFE_ENV}=1. Ship HITL first (planned)."
        )


def _try_refresh_token() -> bool:
    """Silently refresh $ALCF_API_TOKEN using the cached refresh token
    at _TOKENS_PATH. Never prompts; never spawns a subprocess.

    Returns True on success (and updates os.environ["ALCF_API_TOKEN"]),
    False if no cache exists, refresh token is expired, or globus_sdk
    is unavailable. Caller then surfaces a "use alcf_auth to re-auth"
    error to the LLM.

    Same on-disk cache path Level 2 (alcf_auth) writes to, so the two
    paths compose: Level 1 handles access-token expiry (48h) silently;
    Level 2 handles refresh-token expiry (30d) via in-chat browser flow.
    """
    if not os.path.exists(_TOKENS_PATH):
        return False
    try:
        import globus_sdk
        import json
    except ImportError:
        return False
    try:
        with open(_TOKENS_PATH) as f:
            data = json.load(f)
        # Two shapes exist in the wild for the same on-disk path:
        #   flat -- written by _complete_reauth below
        #   nested -- written by ALCF's helper script (globus_sdk.UserApp),
        #             tokens under data["data"]["DEFAULT"][<scope-uuid>]
        refresh_token = data.get("refresh_token")
        if not refresh_token:
            nested = (
                data.get("data", {})
                .get("DEFAULT", {})
                .get(_GLOBUS_SCOPE_CLIENT_ID, {})
            )
            refresh_token = nested.get("refresh_token")
        if not refresh_token:
            return False
        client = globus_sdk.NativeAppAuthClient(_GLOBUS_AUTH_CLIENT_ID)
        response = client.oauth2_refresh_token(refresh_token)
        fresh = response.by_resource_server.get(_GLOBUS_SCOPE_CLIENT_ID)
        if not fresh:
            return False
        # Preserve the refresh_token if the response omits it.
        fresh.setdefault("refresh_token", refresh_token)
        with open(_TOKENS_PATH, "w") as f:
            json.dump(fresh, f, indent=2)
        os.chmod(_TOKENS_PATH, 0o600)
        token = fresh.get("access_token")
        if not token:
            return False
        os.environ["ALCF_API_TOKEN"] = token
        return True
    except Exception:
        return False


def _raise_for_iri(r: "httpx.Response") -> None:
    """Extract the IRI JSON error body when non-2xx, raise a clean
    RuntimeError with just the useful bits. Keeps the LLM from wading
    through a full urllib3 traceback + MDN link."""
    if r.status_code < 400:
        return
    try:
        body = r.json()
        detail = body.get("detail") or body.get("title") or str(body)
    except Exception:
        detail = r.text[:400] or f"<{r.status_code} with no body>"
    # ponytail: strip transient-retry noise so the LLM realises this
    # is a retry-later situation, not a bug in its call.
    if "please try again" in detail.lower() or "RESOURCE_CONFLICT" in detail:
        detail = f"[TRANSIENT, retry in a few seconds] {detail}"
    # 401 after a refresh attempt gets a clearer message so the LLM
    # tells the user to re-auth in a shell instead of retrying forever.
    if r.status_code == 401:
        variant = _detect_variant()
        if variant == "flat":
            hint = (
                "Call alcf_auth_start_reauth (returns a Globus URL for "
                "the user to visit), then alcf_auth_complete_reauth with "
                "the auth code they paste back."
            )
        else:
            hint = (
                "Use the alcf_auth tool: action='start_reauth' returns a "
                "Globus URL for the user to visit, then "
                "action='complete_reauth' with the auth code they paste back."
            )
        detail = f"{detail} -- token expired and silent refresh failed. {hint}"
    raise RuntimeError(f"IRI {r.status_code}: {detail}")


def _with_retry_on_401(fn):
    """Wrap an HTTP call: if it 401s, silently try to refresh the
    token via the on-disk Globus refresh-token flow and retry ONCE.
    If the refresh fails or the retry still 401s, surface the error.
    Any other status is passed through unchanged."""
    def _wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as exc:
            if "IRI 401" not in str(exc):
                raise
            if not _try_refresh_token():
                raise
            return fn(*args, **kwargs)
    return _wrapped


def _get(path: str, *, needs_auth: bool = True, params: dict | None = None) -> Any:
    with httpx.Client(timeout=TIMEOUT_S) as c:
        r = c.get(f"{BASE_URL}{path}", headers=_headers(needs_auth), params=params)
        _raise_for_iri(r)
        return r.json()


def _post(path: str, *, json: dict | None = None) -> Any:
    with httpx.Client(timeout=TIMEOUT_S) as c:
        r = c.post(f"{BASE_URL}{path}", headers=_headers(True), json=json)
        _raise_for_iri(r)
        return r.json()


def _post_with_query(path: str, *, params: dict | None = None) -> Any:
    """POST with query string only (some IRI endpoints like list_jobs
    are POST semantically but take pagination via query string)."""
    with httpx.Client(timeout=TIMEOUT_S) as c:
        r = c.post(f"{BASE_URL}{path}", headers=_headers(True), params=params)
        _raise_for_iri(r)
        return r.json()


def _delete(path: str) -> Any:
    with httpx.Client(timeout=TIMEOUT_S) as c:
        r = c.delete(f"{BASE_URL}{path}", headers=_headers(True))
        _raise_for_iri(r)
        return r.json() if r.content else {"ok": True}


def _delete_with_query(path: str, *, params: dict | None = None) -> Any:
    """DELETE with query params (e.g. filesystem/rm?path=...)."""
    with httpx.Client(timeout=TIMEOUT_S) as c:
        r = c.delete(f"{BASE_URL}{path}", headers=_headers(True), params=params)
        _raise_for_iri(r)
        return r.json() if r.content else {"ok": True}


def _put(path: str, *, json: dict | None = None) -> Any:
    with httpx.Client(timeout=TIMEOUT_S) as c:
        r = c.put(f"{BASE_URL}{path}", headers=_headers(True), json=json)
        _raise_for_iri(r)
        return r.json()


# Wrap all authenticated helpers with the silent 401-refresh-retry.
# _get calls with needs_auth=False (public endpoints) are unaffected --
# the wrapper only kicks on 401, which those never return.
_get = _with_retry_on_401(_get)
_post = _with_retry_on_401(_post)
_post_with_query = _with_retry_on_401(_post_with_query)
_delete = _with_retry_on_401(_delete)
_delete_with_query = _with_retry_on_401(_delete_with_query)
_put = _with_retry_on_401(_put)


# ---------------------------------------------------------------------------
# Interactive re-auth (Level 2). Two-step OAuth Native App flow driven
# via the Globus SDK inline so we never need to shell out during a
# chat session. See docstrings on _start_reauth / _complete_reauth.
# ---------------------------------------------------------------------------


def _detect_variant() -> str:
    """Walk the call stack to see which tool-wrapper module invoked us.

    Returns "flat" when called via alcf_iri_flat_tools, else "category".
    The two variants expose different tool names for the auth-complete
    step, so the next_step string in _start_reauth needs to match.
    """
    import sys
    frame = sys._getframe(1)
    while frame is not None:
        mod = frame.f_globals.get("__name__", "")
        if mod.endswith(".alcf_iri_flat_tools"):
            return "flat"
        if mod.endswith(".alcf_iri_tools"):
            return "category"
        frame = frame.f_back
    return "category"


def _start_reauth() -> dict:
    """Kick off a Globus Native App OAuth flow. Returns the URL the
    user must visit + instructions for the second step.

    Does NOT block. The pending flow is stashed in a module global so
    complete_reauth can finish it when the user pastes the code back.
    """
    global _PENDING_AUTH_CLIENT
    try:
        import globus_sdk
    except ImportError:
        raise RuntimeError(
            "globus_sdk not installed -- pip install globus-sdk in the "
            "Streamlit process's venv to enable in-chat re-auth."
        )
    client = globus_sdk.NativeAppAuthClient(_GLOBUS_AUTH_CLIENT_ID)
    client.oauth2_start_flow(
        requested_scopes=_GLOBUS_SCOPE,
        refresh_tokens=True,
    )
    url = client.oauth2_get_authorize_url()
    _PENDING_AUTH_CLIENT = client
    variant = _detect_variant()
    if variant == "flat":
        next_step = (
            "Ask the user to open this URL in a browser, sign in with "
            "their <user>@alcf.anl.gov identity, copy the resulting "
            "authorization code, and paste it into the next message. "
            "Then call alcf_auth_complete_reauth with auth_code='<the code>'."
        )
    else:
        next_step = (
            "Ask the user to open this URL in a browser, sign in with "
            "their <user>@alcf.anl.gov identity, copy the resulting "
            "authorization code, and paste it into the next message. "
            "Then invoke alcf_auth with action='complete_reauth', "
            "params={'auth_code': '<the code>'}."
        )
    return {
        "status": "pending",
        "url": url,
        "next_step": next_step,
    }


def _complete_reauth(auth_code: str) -> dict:
    """Exchange the auth code from start_reauth for tokens and cache
    them at _TOKENS_PATH so subsequent calls (via the _try_refresh_token
    silent-refresh path) pick them up automatically."""
    global _PENDING_AUTH_CLIENT
    if _PENDING_AUTH_CLIENT is None:
        raise RuntimeError(
            "No pending re-auth. Call action='start_reauth' first."
        )
    if not auth_code or not auth_code.strip():
        raise RuntimeError("auth_code is empty.")
    try:
        response = _PENDING_AUTH_CLIENT.oauth2_exchange_code_for_tokens(
            auth_code.strip(),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Globus rejected the auth code: {exc}. Ask the user to "
            "restart with action='start_reauth' and try again."
        )
    _PENDING_AUTH_CLIENT = None
    data = response.by_resource_server.get(_GLOBUS_SCOPE_CLIENT_ID)
    if not data:
        raise RuntimeError(
            "Globus returned tokens but not for the IRI resource server. "
            "The Globus account may not be linked to ALCF."
        )
    # Cache to disk in the same location _try_refresh_token reads from,
    # so future calls hit the silent-refresh path and never re-prompt.
    import json
    os.makedirs(os.path.dirname(_TOKENS_PATH), exist_ok=True)
    with open(_TOKENS_PATH, "w") as f:
        json.dump(data, f, indent=2)
    os.chmod(_TOKENS_PATH, 0o600)
    # Also update the in-process env so the NEXT call in this same turn
    # uses the fresh token without a subprocess round-trip.
    token = data.get("access_token")
    if token:
        os.environ["ALCF_API_TOKEN"] = token
    return {"status": "ok", "message": "re-authenticated; retry your query."}


# ---------------------------------------------------------------------------
# Action tables per category. Keys are action names, values are:
#   (kind, description, params_schema, invoker)
# where invoker(**kwargs) returns the endpoint result.
#
# params_schema: a dict of {name: (type_str, required_bool, description)}
#   Rendered on-demand for the LLM instead of shipped upfront.
# kind: "read" | "write" -- write actions get gated behind $ALCF_IRI_ALLOW_UNSAFE.
# ---------------------------------------------------------------------------


def _facility_actions():
    return {
        "get": (
            "read",
            "Facility-wide metadata (name, sites).",
            {},
            lambda **_: _get("/facility", needs_auth=False),
        ),
        "list_sites": (
            "read",
            "List all sites under this facility.",
            {},
            lambda **_: _get("/facility/sites", needs_auth=False),
        ),
        "get_site": (
            "read",
            "Get one site's details.",
            {"site_id": ("str", True, "Site UUID.")},
            lambda site_id, **_: _get(f"/facility/sites/{site_id}", needs_auth=False),
        ),
    }


def _status_actions():
    return {
        "list_resources": (
            "read",
            "List all resources (compute/storage/service). Optional filters.",
            {
                "resource_type": ("str", False, "'compute' | 'storage' | 'service'"),
                "current_status": ("str", False, "'up' | 'down' | 'maintenance' | ..."),
                "group": ("str", False, "e.g. 'computes', 'storages'"),
            },
            lambda **kw: _get("/status/resources", needs_auth=False, params={
                k: v for k, v in kw.items() if v is not None
            }),
        ),
        "get_resource": (
            "read",
            "Get one resource's current state (up/down/maintenance/degraded).",
            {"name": ("str", True, "Resource name like 'Aurora', or its UUID.")},
            lambda name, **_: _get(f"/status/resources/{_resource_id(name)}", needs_auth=False),
        ),
        "list_incidents": (
            "read",
            "List all incidents (outages, scheduled maintenance).",
            {},
            lambda **_: _get("/status/incidents", needs_auth=False),
        ),
        "get_incident": (
            "read",
            "Get one incident and its events.",
            {"incident_id": ("str", True, "Incident UUID.")},
            lambda incident_id, **_: _get(f"/status/incidents/{incident_id}", needs_auth=False),
        ),
        "list_events": (
            "read",
            "List all events (state changes on resources).",
            {},
            lambda **_: _get("/status/events", needs_auth=False),
        ),
        "get_event": (
            "read",
            "Get one event.",
            {"event_id": ("str", True, "Event UUID.")},
            lambda event_id, **_: _get(f"/status/events/{event_id}", needs_auth=False),
        ),
    }


def _account_actions():
    return {
        "list_capabilities": (
            "read",
            "List capabilities the facility exposes (public).",
            {},
            lambda **_: _get("/account/capabilities", needs_auth=False),
        ),
        "get_capability": (
            "read",
            "One capability's details.",
            {"capability_id": ("str", True, "Capability UUID.")},
            lambda capability_id, **_: _get(
                f"/account/capabilities/{capability_id}", needs_auth=False,
            ),
        ),
        "list_projects": (
            "read",
            "Your projects. Needs $ALCF_API_TOKEN.",
            {},
            lambda **_: _get("/account/projects"),
        ),
        "get_project": (
            "read",
            "One project's details.",
            {"project_id": ("str", True, "Project UUID.")},
            lambda project_id, **_: _get(f"/account/projects/{project_id}"),
        ),
        "list_allocations": (
            "read",
            "All allocations under one project (balance, charged, jobs).",
            {"project_id": ("str", True, "Project UUID.")},
            lambda project_id, **_: _get(
                f"/account/projects/{project_id}/project_allocations",
            ),
        ),
        "get_allocation": (
            "read",
            "One project allocation's detail (per-machine balance).",
            {
                "project_id": ("str", True, "Project UUID."),
                "allocation_id": ("str", True, "Allocation UUID."),
            },
            lambda project_id, allocation_id, **_: _get(
                f"/account/projects/{project_id}/project_allocations/{allocation_id}",
            ),
        ),
        "list_user_allocations": (
            "read",
            "Per-user slice of a project allocation.",
            {
                "project_id": ("str", True, "Project UUID."),
                "allocation_id": ("str", True, "Allocation UUID."),
            },
            lambda project_id, allocation_id, **_: _get(
                f"/account/projects/{project_id}/project_allocations/{allocation_id}/user_allocations",
            ),
        ),
    }


def _compute_actions():
    return {
        "get_job_status": (
            "read",
            "Status of one PBS job by id. Set historical=true to look "
            "up jobs that already finished.",
            {
                "machine": ("str", True, "'crux' | 'aurora' | 'polaris' or UUID."),
                "job_id": ("str", True, "PBS job id (numeric or full form)."),
                "historical": ("bool", False,
                    "Search historical/completed jobs instead of active queue."),
            },
            lambda machine, job_id, historical=None, **_: _get(
                f"/compute/status/{_resource_id(machine)}/{job_id}",
                params={"historical": historical} if historical is not None else None,
            ),
        ),
        "list_jobs": (
            "read",
            "List jobs on one machine with pagination. Set "
            "historical=true for completed jobs. NOTE: this is a POST "
            "in the API even though it's read-only.",
            {
                "machine": ("str", True, "Machine name or UUID."),
                "historical": ("bool", False,
                    "True = completed jobs, False = active (default)."),
                "limit": ("int", False, "Max results to return."),
                "offset": ("int", False, "Pagination offset."),
            },
            lambda machine, historical=None, limit=None, offset=None, **_: (
                _post_with_query(
                    f"/compute/status/{_resource_id(machine)}",
                    params={
                        k: v for k, v in {
                            "historical": historical, "limit": limit, "offset": offset,
                        }.items() if v is not None
                    },
                )
            ),
        ),
        "submit_job": (
            "write",
            "Submit a PBS job via a PSI/J-flavored JobSpec. Returns "
            "the PBS job record directly: {id: '<pbs_id>.<host>', "
            "status: {state: 'queued'|'active'|..., exit_code: 0}}. "
            "Use alcf_compute action='get_job_status' with that id to "
            "watch state transitions. UNSAFE at the tool layer; will "
            f"raise RuntimeError if ${UNSAFE_ENV}=1 is not set server-"
            "side -- attempt normally, do NOT refuse preemptively. "
            "EXACT jobspec body (from ALCF docs verbatim): {"
            "\"executable\": \"/bin/bash\", "
            "\"arguments\": [\"-lc\", \"echo hello; sleep 10\"], "
            "\"name\": \"my_job\", "
            "\"stdout_path\": \"/home/<user>/out\", "
            "\"stderr_path\": \"/home/<user>/err\", "
            "\"resources\": {\"node_count\": 1}, "
            "\"attributes\": {"
            "\"duration\": 300, "
            "\"queue_name\": \"debug\", "
            "\"account\": \"<project>\", "
            "\"custom_attributes\": {\"filesystems\": \"home:eagle\"}"
            "}} -- NOTE `filesystems` is a COLON-SEPARATED STRING "
            "(e.g. \"home:eagle\"), NOT a list.",
            {
                "machine": ("str", True, "Machine name or UUID."),
                "jobspec": ("dict", True,
                    "Full JobSpec body -- see description for keys."),
            },
            lambda machine, jobspec, **_: (
                _check_unsafe("submit_job") or _post(
                    f"/compute/job/{_resource_id(machine)}", json=jobspec,
                )
            ),
        ),
        "update_job": (
            "write",
            "Update fields of an already-scheduled job. UNSAFE. Only "
            "some fields are updatable; facility-dependent which ones.",
            {
                "machine": ("str", True, "Machine name or UUID."),
                "job_id": ("str", True, "PBS job id."),
                "jobspec": ("dict", True, "Partial JobSpec with updated fields."),
            },
            lambda machine, job_id, jobspec, **_: (
                _check_unsafe("update_job") or _put(
                    f"/compute/job/{_resource_id(machine)}/{job_id}", json=jobspec,
                )
            ),
        ),
        "cancel_job": (
            "write",
            f"Cancel/qdel a job. UNSAFE. Needs ${UNSAFE_ENV}=1.",
            {
                "machine": ("str", True, "Machine name or UUID."),
                "job_id": ("str", True, "PBS job id."),
            },
            lambda machine, job_id, **_: (
                _check_unsafe("cancel_job") or _delete(
                    f"/compute/cancel/{_resource_id(machine)}/{job_id}",
                )
            ),
        ),
    }


def _filesystem_actions():
    def _r(name):
        return _resource_id(name)
    # ALCF's fs endpoints target either compute-resource UUIDs (Crux,
    # Aurora, Polaris) OR storage-resource UUIDs (Eagle, Home). The docs
    # examples use storage UUIDs. Either works; agent picks based on
    # which filesystem it wants to touch.
    return {
        "ls": (
            "read",
            "List directory contents.",
            {
                "machine": ("str", True,
                    "STORAGE resource, NOT the compute machine. Pick "
                    "by path prefix: /eagle/... -> 'eagle', /home/... "
                    "-> 'home'. Do NOT pass 'aurora', 'crux', or "
                    "'polaris' here -- IRI's filesystem endpoints "
                    "target storage UUIDs, not compute UUIDs. "
                    "/flare/... is not exposed."),
                "path": ("str", True, "Absolute path to list."),
                "show_hidden": ("bool", False, "Include dotfiles."),
                "recursive": ("bool", False, "Recurse into subdirs."),
                "dereference": ("bool", False, "Follow symlinks."),
            },
            lambda machine, path, show_hidden=False, recursive=False,
                   dereference=False, **_: _get(
                f"/filesystem/ls/{_r(machine)}",
                params={"path": path, "showHidden": show_hidden,
                        "recursive": recursive, "dereference": dereference},
            ),
        ),
        "stat": (
            "read",
            "File metadata (size, mode, mtime). NOTE: ALCF returned "
            "501 'not implemented yet' as of 2026-08-11 -- use `ls` "
            "which returns similar per-entry metadata.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path.")},
            lambda machine, path, **_: _get(
                f"/filesystem/stat/{_r(machine)}", params={"path": path},
            ),
        ),
        "cat": (
            "read",
            "Read full file contents (use `view` for large files with pagination).",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path.")},
            lambda machine, path, **_: _get(
                f"/filesystem/file/{_r(machine)}", params={"path": path},
            ),
        ),
        "view": (
            "read",
            "Paginated file read. Returns up to `size` bytes starting "
            "at `offset`. Preferred over `cat` for large files.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path."),
             "size": ("int", False, "Max bytes to return (facility default if omitted)."),
             "offset": ("int", False, "Byte offset to start from (default 0).")},
            lambda machine, path, size=None, offset=None, **_: _get(
                f"/filesystem/view/{_r(machine)}",
                params={
                    k: v for k, v in {
                        "path": path, "size": size, "offset": offset,
                    }.items() if v is not None
                },
            ),
        ),
        "head": (
            "read",
            "First N lines of a file.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path."),
             "lines": ("int", False, "Number of lines (facility default if omitted).")},
            lambda machine, path, lines=None, **_: _get(
                f"/filesystem/head/{_r(machine)}",
                params={
                    k: v for k, v in {
                        "path": path, "lines": lines,
                    }.items() if v is not None
                },
            ),
        ),
        "tail": (
            "read",
            "Last N lines of a file (useful for job stderr/stdout).",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path."),
             "lines": ("int", False, "Number of lines.")},
            lambda machine, path, lines=None, **_: _get(
                f"/filesystem/tail/{_r(machine)}",
                params={
                    k: v for k, v in {
                        "path": path, "lines": lines,
                    }.items() if v is not None
                },
            ),
        ),
        "checksum": (
            "read",
            "MD5/SHA of a file.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path.")},
            lambda machine, path, **_: _get(
                f"/filesystem/checksum/{_r(machine)}", params={"path": path},
            ),
        ),
        "download": (
            "read",
            "Download a file to the caller (returns bytes; small files only).",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path.")},
            lambda machine, path, **_: _get(
                f"/filesystem/download/{_r(machine)}", params={"path": path},
            ),
        ),
        "mkdir": (
            "write",
            "Create directory. UNSAFE.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path."),
             "parent": ("bool", False,
                 "If true, create parent dirs (like mkdir -p). Default false.")},
            lambda machine, path, parent=False, **_: (
                _check_unsafe("mkdir") or _post(
                    f"/filesystem/mkdir/{_r(machine)}",
                    json={"path": path, "parent": parent},
                )
            ),
        ),
        "rm": (
            "write",
            "Remove file/dir. UNSAFE.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path.")},
            lambda machine, path, **_: (
                _check_unsafe("rm") or _delete_with_query(
                    f"/filesystem/rm/{_r(machine)}", params={"path": path},
                )
            ),
        ),
        "chmod": (
            "write",
            "Change permissions. UNSAFE. mode is octal string like '700'.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path."),
             "mode": ("str", True, "Octal mode like '700' or '755'.")},
            lambda machine, path, mode, **_: (
                _check_unsafe("chmod") or _put(
                    f"/filesystem/chmod/{_r(machine)}",
                    json={"path": path, "mode": mode},
                )
            ),
        ),
        "chown": (
            "write",
            "Change owner/group. UNSAFE.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "path": ("str", True, "Absolute path."),
             "owner": ("str", False, "New owner username or uid."),
             "group": ("str", False, "New group name or gid.")},
            lambda machine, path, owner=None, group=None, **_: (
                _check_unsafe("chown") or _put(
                    f"/filesystem/chown/{_r(machine)}",
                    json={"path": path, **({"owner": owner} if owner else {}),
                          **({"group": group} if group else {})},
                )
            ),
        ),
        # The following actions exist in the OpenAPI spec but ALCF has
        # NOT published examples for them. Body shapes below are best
        # guesses from the spec's schema references -- may return 400
        # if ALCF expects different keys. Left in so the LLM can
        # discover them; agent should prefer scp/ssh if these 400.
        "mv": (
            "write",
            "Rename/move. UNSAFE. Body shape is unofficial (ALCF has "
            "not documented an example).",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "src": ("str", True, "Source path."),
             "dst": ("str", True, "Destination path.")},
            lambda machine, src, dst, **_: (
                _check_unsafe("mv") or _post(
                    f"/filesystem/mv/{_r(machine)}",
                    json={"source": src, "destination": dst},
                )
            ),
        ),
        "cp": (
            "write",
            "Copy. UNSAFE. Body shape unofficial.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "src": ("str", True, "Source path."),
             "dst": ("str", True, "Destination path.")},
            lambda machine, src, dst, **_: (
                _check_unsafe("cp") or _post(
                    f"/filesystem/cp/{_r(machine)}",
                    json={"source": src, "destination": dst},
                )
            ),
        ),
        "symlink": (
            "write",
            "Create symlink. UNSAFE. Body shape unofficial.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "target": ("str", True, "Existing path being linked to."),
             "link_path": ("str", True, "New symlink path.")},
            lambda machine, target, link_path, **_: (
                _check_unsafe("symlink") or _post(
                    f"/filesystem/symlink/{_r(machine)}",
                    json={"target": target, "link_path": link_path},
                )
            ),
        ),
        "compress": (
            "write",
            "tar/gzip files. UNSAFE. Body shape unofficial.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "paths": ("list[str]", True, "Absolute paths to include."),
             "archive": ("str", True, "Output archive path.")},
            lambda machine, paths, archive, **_: (
                _check_unsafe("compress") or _post(
                    f"/filesystem/compress/{_r(machine)}",
                    json={"paths": paths, "archive": archive},
                )
            ),
        ),
        "extract": (
            "write",
            "Untar. UNSAFE. Body shape unofficial.",
            {"machine": ("str", True,
                "STORAGE resource, NOT the compute machine. Pick by "
                "path prefix: /eagle/... -> 'eagle', /home/... -> "
                "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' "
                "here -- IRI's filesystem endpoints target storage "
                "UUIDs, not compute UUIDs. /flare/... is not exposed."),
             "archive": ("str", True, "Archive path."),
             "dest": ("str", True, "Destination directory.")},
            lambda machine, archive, dest, **_: (
                _check_unsafe("extract") or _post(
                    f"/filesystem/extract/{_r(machine)}",
                    json={"archive": archive, "destination": dest},
                )
            ),
        ),
    }


def _auth_actions():
    return {
        "start_reauth": (
            "read",  # not a write op -- doesn't change ALCF state
            "Begin an interactive Globus re-auth flow. Returns a URL "
            "the user must visit + instructions for step 2.",
            {},
            lambda **_: _start_reauth(),
        ),
        "complete_reauth": (
            "read",
            "Finish the re-auth flow with the auth code the user pasted.",
            {"auth_code": ("str", True,
                "The 30-char code Globus displayed after the user "
                "signed in via the start_reauth URL.")},
            lambda auth_code, **_: _complete_reauth(auth_code),
        ),
    }


def _task_actions():
    return {
        "list": (
            "read",
            "List all async task handles owned by the caller.",
            {},
            lambda **_: _get("/task"),
        ),
        "get": (
            "read",
            "Get one task's state (many FS/compute ops return a task_id).",
            {"task_id": ("str", True, "Task UUID.")},
            lambda task_id, **_: _get(f"/task/{task_id}"),
        ),
        "cancel": (
            "write",
            "Cancel an in-flight task. UNSAFE.",
            {"task_id": ("str", True, "Task UUID.")},
            lambda task_id, **_: (
                _check_unsafe("task.cancel") or _delete(f"/task/{task_id}")
            ),
        ),
    }


CATEGORIES: dict[str, dict[str, tuple]] = {
    "facility": _facility_actions(),
    "status": _status_actions(),
    "account": _account_actions(),
    "compute": _compute_actions(),
    "filesystem": _filesystem_actions(),
    "task": _task_actions(),
    "auth": _auth_actions(),
}


def dispatch(category: str, action: str, params: dict[str, Any]) -> Any:
    """Route a category+action pair to the right invoker.

    Three action kinds:
      - "list_actions" -> list of {name: {kind, description}} for the category
      - "describe"     -> full schema for one action; requires
                          params={"target_action": <name>}
      - <name>         -> invoke that action with the supplied params

    Raises ValueError for unknown actions, RuntimeError for API errors
    (bubbled up from _raise_for_iri).
    """
    actions = CATEGORIES[category]
    if action == "list_actions":
        return {
            name: {"kind": spec[0], "description": spec[1]}
            for name, spec in actions.items()
        }
    if action == "describe":
        target = params.get("target_action")
        if target not in actions:
            raise ValueError(
                f"Unknown action {target!r} for category {category!r}. "
                f"Available: {sorted(actions)}"
            )
        kind, desc, params_schema, _ = actions[target]
        return {
            "action": target,
            "kind": kind,
            "description": desc,
            "params": {
                p: {"type": t, "required": req, "description": d}
                for p, (t, req, d) in params_schema.items()
            },
        }
    if action not in actions:
        raise ValueError(
            f"Unknown action {action!r} for category {category!r}. "
            f"Available: {sorted(actions)}. Call action='list_actions' "
            "to see all, or action='describe' target_action=<name> for one."
        )
    return actions[action][3](**params)
