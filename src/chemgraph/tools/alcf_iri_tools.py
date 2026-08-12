"""LangChain ``@tool`` wrappers for the ALCF IRI Facility API.

Each tool delegates to the pure-Python implementation in
:mod:`chemgraph.tools.alcf_iri_core`.

Seven category tools (facility, status, account, compute, filesystem,
task, auth) instead of one @tool per endpoint. Each accepts an
``action`` enum plus optional ``params`` dict, and uses the discovery
protocol (action='list_actions' | 'describe' | <name>) to keep the
LLM's initial prompt schema small. See the module docstring in
alcf_iri_core.py for the full design rationale.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from chemgraph.tools.alcf_iri_core import (
    # Re-export so existing `from alcf_iri_tools import ...` keeps working
    CATEGORIES,
    dispatch,
)

__all__ = [
    "CATEGORIES",
    "dispatch",
    "alcf_facility",
    "alcf_status",
    "alcf_account",
    "alcf_compute",
    "alcf_filesystem",
    "alcf_task",
    "alcf_auth",
    "ALCF_IRI_TOOLS",
]


_DISCOVERY_HINT = (
    " Discovery: call action='list_actions' to see the available actions "
    "on this tool, then action='describe' with params={'target_action':"
    "<name>} to see the schema for one, then invoke with action=<name>, "
    "params={...}. Do not guess action names."
)


@tool(description=(
    "ALCF Facility API — facility and site metadata (what facility this "
    "is, which physical sites belong to it). No auth required."
    + _DISCOVERY_HINT
))
def alcf_facility(action: str, params: dict | None = None) -> Any:
    return dispatch("facility", action, params or {})


@tool(description=(
    "ALCF Facility API — real-time state of ALCF resources (compute, "
    "storage, services), plus incidents and events. Use for questions "
    "about whether a machine is up, planned outages, or state history. "
    "No auth required."
    + _DISCOVERY_HINT
))
def alcf_status(action: str, params: dict | None = None) -> Any:
    return dispatch("status", action, params or {})


@tool(description=(
    "ALCF Facility API — accounts, projects, allocations, per-user "
    "quotas, and the catalog of capabilities the facility exposes. Use "
    "for questions about who you are, what projects you belong to, and "
    "how much allocation you have. Most actions need $ALCF_API_TOKEN."
    + _DISCOVERY_HINT
))
def alcf_account(action: str, params: dict | None = None) -> Any:
    return dispatch("account", action, params or {})


@tool(description=(
    "ALCF Facility API — PBS batch job control on ALCF machines. "
    "Read actions (get_job_status, list_jobs) work unconditionally. "
    "Write actions (submit_job, update_job, cancel_job) will succeed "
    "if the server is configured to permit them and raise a clear "
    "RuntimeError otherwise -- ALWAYS ATTEMPT them if the user asks; "
    "do NOT refuse preemptively. Needs $ALCF_API_TOKEN."
    + _DISCOVERY_HINT
))
def alcf_compute(action: str, params: dict | None = None) -> Any:
    return dispatch("compute", action, params or {})


@tool(description=(
    "ALCF Facility API — remote filesystem ops on ALCF storage via "
    "HTTPS. Read actions (ls, stat, cat, head, tail, checksum, "
    "download) work unconditionally. Write actions (mkdir, rm, chmod, "
    "...) will succeed if the server is configured to permit them and "
    "raise a clear RuntimeError otherwise -- ALWAYS ATTEMPT them if "
    "the user asks; do NOT refuse preemptively. Needs $ALCF_API_TOKEN. "
    "IMPORTANT: filesystem targets STORAGE resources, NOT compute. "
    "Pick the `machine` argument by path prefix: /eagle/... -> "
    "'eagle', /home/... -> 'home'. Do NOT pass 'aurora', 'crux', or "
    "'polaris' -- those are compute UUIDs and IRI's filesystem "
    "endpoints will 400. /flare/... is not exposed by IRI at all; "
    "tell the user to use scp for /flare paths."
    + _DISCOVERY_HINT
))
def alcf_filesystem(action: str, params: dict | None = None) -> Any:
    return dispatch("filesystem", action, params or {})


@tool(description=(
    "ALCF Facility API — handles for asynchronous operations returned "
    "by other endpoints (long-running filesystem or compute ops give "
    "you a task_id to poll). Needs $ALCF_API_TOKEN."
    + _DISCOVERY_HINT
))
def alcf_task(action: str, params: dict | None = None) -> Any:
    return dispatch("task", action, params or {})


@tool(description=(
    "ALCF Facility API — interactive Globus re-auth. Use this ONLY "
    "when another alcf_* tool returned a 401 that mentioned expired "
    "token AND silent refresh failed. Two-step: (1) call "
    "action='start_reauth' to get a URL for the user to visit, show "
    "them the URL and ask for the auth code they get after signing "
    "in; (2) call action='complete_reauth' with params={'auth_code': "
    "'<the code>'}. After success, retry the original query."
    + _DISCOVERY_HINT
))
def alcf_auth(action: str, params: dict | None = None) -> Any:
    return dispatch("auth", action, params or {})


ALCF_IRI_TOOLS = [
    alcf_facility, alcf_status, alcf_account,
    alcf_compute, alcf_filesystem, alcf_task, alcf_auth,
]
