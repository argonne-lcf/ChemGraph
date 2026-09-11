"""MCP server exposing ALCF's IRI Facility API.

Wraps the same :mod:`chemgraph.tools.alcf_iri_core` implementation used
by the ``single_agent_iri`` LangGraph workflow, but presents it via the
Model Context Protocol so any MCP-speaking client (Claude Desktop,
another agent framework, main_agent's MCP wiring, etc.) can reach it.

Two tool-registration variants, chosen at launch time:

* ``flat`` (default) -- 43 tools, one per IRI action, named
  ``alcf_<category>_<action>``. Higher judge score on our eval; matches
  :mod:`chemgraph.tools.alcf_iri_flat_tools`.
* ``category`` -- 7 dispatcher tools (facility, status, account,
  compute, filesystem, task, auth), each taking ``action: str`` + optional
  ``params: dict``, with the standard ``list_actions``/``describe``
  discovery protocol. Smaller upfront schema surface; matches
  :mod:`chemgraph.tools.alcf_iri_tools`.

Select with ``--variant flat|category`` or ``$CHEMGRAPH_IRI_MCP_VARIANT``.
CLI flag wins; env var is the fallback (useful for MCP client configs
that can pass env but not argv, e.g. Claude Desktop).

Auth follows the same rules regardless of variant: reads
``$ALCF_API_TOKEN``, falls back to the on-disk Globus refresh cache,
and exposes interactive re-auth via ``alcf_auth_*`` tools (flat) or
``alcf_auth(action='start_reauth' | 'complete_reauth')`` (category).

Write actions (submit_job, cancel_job, rm, mkdir, chmod, ...) require
``$ALCF_IRI_ALLOW_UNSAFE=1`` and raise RuntimeError otherwise.

Usage:
    python -m chemgraph.mcp.alcf_iri_mcp
    python -m chemgraph.mcp.alcf_iri_mcp --variant category
    python -m chemgraph.mcp.alcf_iri_mcp --transport streamable_http --port 9010
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

from chemgraph.tools.alcf_iri_core import CATEGORIES, dispatch


mcp = FastMCP(
    name="ChemGraph ALCF IRI Tools",
    instructions="""
        You expose ALCF Facility API tools (https://api.alcf.anl.gov)
        for querying and managing HPC resources at Argonne Leadership
        Computing Facility.

        Two shapes exist depending on how this server was launched:

        FLAT (default): one tool per action, named
        alcf_<category>_<action>. Call by name with the action's args.

        CATEGORY: seven dispatcher tools (alcf_facility, alcf_status,
        alcf_account, alcf_compute, alcf_filesystem, alcf_task,
        alcf_auth). Each takes action: str + params: dict. Discover
        available actions with action='list_actions', then get one
        action's schema with action='describe',
        params={'target_action': <name>}, then invoke with
        action=<name>, params={...}. Do not guess action names.

        Common guidelines (both variants):
        - Public endpoints (facility, status, most account/capability
          calls) work without authentication.
        - Authenticated endpoints (projects, allocations, PBS jobs,
          filesystem) need $ALCF_API_TOKEN or a cached Globus refresh
          token. On 401 the error message will tell you which auth
          tools to call.
        - Write actions (submit_job, cancel_job, mkdir, rm, chmod, ...)
          are gated behind $ALCF_IRI_ALLOW_UNSAFE=1 at the tool layer.
          Attempt them normally and surface any RuntimeError -- do NOT
          refuse preemptively.
        - Machine names are case-insensitive ('aurora', 'crux', 'polaris').
          Filesystem tools take STORAGE resource names ('eagle', 'home'),
          not the compute machine.
        - Prefer historical=true on list_jobs / get_job_status when the
          user asks about completed jobs; the default is the live queue.
    """,
)


_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "bool": bool,
    "dict": dict,
    "list[str]": list,
}


# ── Flat variant ────────────────────────────────────────────────────────


def _build_flat_wrapper(
    category: str,
    action: str,
    kind: str,
    description: str,
    params_schema: dict,
    invoker: Callable,
) -> tuple[Callable, str, str]:
    """Build one alcf_<category>_<action> MCP tool wrapper.

    Signature carries the action's declared params so FastMCP can
    derive a JSON schema from it. Required params become
    positional-or-keyword parameters without defaults; optional params
    default to ``None`` with ``Optional[...]`` annotations so FastMCP
    marks them non-required.
    """
    tool_name = f"alcf_{category}_{action}"

    parameters: list[inspect.Parameter] = []
    for pname, (type_str, required, _desc) in params_schema.items():
        py_type = _TYPE_MAP.get(type_str, str)
        if required:
            annotation = py_type
            default = inspect.Parameter.empty
        else:
            annotation = Optional[py_type]
            default = None
        parameters.append(
            inspect.Parameter(
                pname,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=annotation,
            )
        )

    def wrapper(**kwargs: Any) -> Any:
        # Drop None values so the invoker's `if v is not None` filters
        # behave the same as they do under the LangChain wrappers.
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return invoker(**cleaned)

    wrapper.__name__ = tool_name
    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=parameters,
        return_annotation=Any,
    )

    tag = "read" if kind == "read" else "write"
    return wrapper, tool_name, f"[{tag}] {description}"


def _register_flat() -> int:
    """Register 43 tools, one per (category, action) in CATEGORIES."""
    count = 0
    for category, actions in CATEGORIES.items():
        for action, (kind, desc, params_schema, invoker) in actions.items():
            fn, name, description = _build_flat_wrapper(
                category, action, kind, desc, params_schema, invoker,
            )
            mcp.add_tool(fn, name=name, description=description)
            count += 1
    return count


# ── Category variant ────────────────────────────────────────────────────


_DISCOVERY_HINT = (
    " Discovery: call action='list_actions' to see the available "
    "actions on this tool, then action='describe' with "
    "params={'target_action': <name>} to see the schema for one, then "
    "invoke with action=<name>, params={...}. Do not guess action names."
)


_CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "facility": (
        "ALCF Facility API -- facility and site metadata (what facility "
        "this is, which physical sites belong to it). No auth required."
    ),
    "status": (
        "ALCF Facility API -- real-time state of ALCF resources "
        "(compute, storage, services), plus incidents and events. Use "
        "for questions about whether a machine is up, planned outages, "
        "or state history. No auth required."
    ),
    "account": (
        "ALCF Facility API -- accounts, projects, allocations, "
        "per-user quotas, and the catalog of capabilities the facility "
        "exposes. Most actions need $ALCF_API_TOKEN."
    ),
    "compute": (
        "ALCF Facility API -- PBS batch job control on ALCF machines. "
        "Read actions (get_job_status, list_jobs) work unconditionally. "
        "Write actions (submit_job, update_job, cancel_job) will "
        "succeed if the server is configured to permit them and raise "
        "a clear RuntimeError otherwise -- ALWAYS ATTEMPT them if the "
        "user asks; do NOT refuse preemptively. Needs $ALCF_API_TOKEN."
    ),
    "filesystem": (
        "ALCF Facility API -- remote filesystem ops on ALCF storage "
        "via HTTPS. Read actions (ls, stat, cat, head, tail, checksum, "
        "download) work unconditionally. Write actions (mkdir, rm, "
        "chmod, ...) will succeed if the server is configured to "
        "permit them and raise a clear RuntimeError otherwise -- "
        "ALWAYS ATTEMPT them if the user asks; do NOT refuse "
        "preemptively. Needs $ALCF_API_TOKEN. IMPORTANT: filesystem "
        "targets STORAGE resources, NOT compute. Pick the `machine` "
        "argument by path prefix: /eagle/... -> 'eagle', /home/... -> "
        "'home'. Do NOT pass 'aurora', 'crux', or 'polaris' -- those "
        "are compute UUIDs and IRI's filesystem endpoints will 400. "
        "/flare/... is not exposed by IRI at all; tell the user to use "
        "scp for /flare paths."
    ),
    "task": (
        "ALCF Facility API -- handles for asynchronous operations "
        "returned by other endpoints (long-running filesystem or "
        "compute ops give you a task_id to poll). Needs $ALCF_API_TOKEN."
    ),
    "auth": (
        "ALCF Facility API -- interactive Globus re-auth. Use this "
        "ONLY when another alcf_* tool returned a 401 that mentioned "
        "expired token AND silent refresh failed. Two-step: (1) call "
        "action='start_reauth' to get a URL for the user to visit, "
        "show them the URL and ask for the auth code they get after "
        "signing in; (2) call action='complete_reauth' with "
        "params={'auth_code': '<the code>'}. After success, retry the "
        "original query."
    ),
}


def _build_category_dispatcher(category: str) -> Callable:
    """Build one alcf_<category> dispatcher for the category variant."""
    def wrapper(
        action: str,
        params: Optional[dict] = None,
    ) -> Any:
        return dispatch(category, action, params or {})

    wrapper.__name__ = f"alcf_{category}"
    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=[
            inspect.Parameter(
                "action",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            ),
            inspect.Parameter(
                "params",
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=Optional[dict],
            ),
        ],
        return_annotation=Any,
    )
    return wrapper


def _register_category() -> int:
    """Register 7 dispatcher tools -- one per category."""
    count = 0
    for category in CATEGORIES.keys():
        fn = _build_category_dispatcher(category)
        description = _CATEGORY_DESCRIPTIONS.get(category, "") + _DISCOVERY_HINT
        mcp.add_tool(fn, name=f"alcf_{category}", description=description)
        count += 1
    return count


# ── Variant selection ──────────────────────────────────────────────────


VALID_VARIANTS = ("flat", "category")


def register(variant: str = "flat") -> int:
    """Register the chosen tool set on the module-level ``mcp`` server.

    Callable from both the ``__main__`` path (CLI-driven) and from
    embedders that ``import chemgraph.mcp.alcf_iri_mcp`` and want to
    pick a variant themselves. Returns the number of tools registered.
    """
    if variant == "flat":
        return _register_flat()
    if variant == "category":
        return _register_category()
    raise ValueError(
        f"Unknown IRI MCP variant: {variant!r} (expected one of {VALID_VARIANTS})"
    )


def _resolve_variant() -> str:
    """Pull --variant off sys.argv so run_mcp_server sees only its own flags.

    Precedence: CLI --variant > $CHEMGRAPH_IRI_MCP_VARIANT > "flat".
    """
    import argparse
    import sys

    env_default = os.environ.get("CHEMGRAPH_IRI_MCP_VARIANT", "flat")
    if env_default not in VALID_VARIANTS:
        raise ValueError(
            f"CHEMGRAPH_IRI_MCP_VARIANT={env_default!r} invalid "
            f"(expected one of {VALID_VARIANTS})"
        )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--variant", choices=VALID_VARIANTS, default=env_default)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.variant


# Register the default variant at import time so ``from chemgraph.mcp
# import alcf_iri_mcp`` gives an already-populated server (embedders
# who want the other variant can call ``register('category')`` on a
# fresh FastMCP instance, or set $CHEMGRAPH_IRI_MCP_VARIANT before
# import). The CLI path below re-parses --variant and only re-registers
# if it differs from what import time chose.
_INITIAL_VARIANT = os.environ.get("CHEMGRAPH_IRI_MCP_VARIANT", "flat")
if _INITIAL_VARIANT not in VALID_VARIANTS:
    raise ValueError(
        f"CHEMGRAPH_IRI_MCP_VARIANT={_INITIAL_VARIANT!r} invalid "
        f"(expected one of {VALID_VARIANTS})"
    )
_TOOL_COUNT = register(_INITIAL_VARIANT)


if __name__ == "__main__":
    _cli_variant = _resolve_variant()
    if _cli_variant != _INITIAL_VARIANT:
        # Rebuild the tool set. FastMCP's tool store isn't part of its
        # public API so we swap in a fresh manager rather than mutate.
        from mcp.server.fastmcp.tools import ToolManager

        mcp._tool_manager = ToolManager()  # type: ignore[attr-defined]
        _TOOL_COUNT = register(_cli_variant)

    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9010)
