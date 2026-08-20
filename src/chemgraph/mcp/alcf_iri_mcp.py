"""MCP server exposing ALCF's IRI Facility API as flat @mcp tools.

Wraps the same :mod:`chemgraph.tools.alcf_iri_core` implementation used
by the ``single_agent_iri`` LangGraph workflow, but presents it via the
Model Context Protocol so any MCP-speaking client (Claude Desktop,
another agent framework, main_agent's MCP wiring, etc.) can reach it.

Registers one tool per IRI action (~43 total) with names
``alcf_<category>_<action>`` -- mirrors
:mod:`chemgraph.tools.alcf_iri_flat_tools` so behaviour is identical.

Auth follows the same rules as the LangChain flat tools: reads
``$ALCF_API_TOKEN``, falls back to the on-disk Globus refresh cache,
and exposes ``alcf_auth_start_reauth`` / ``alcf_auth_complete_reauth``
for the interactive OAuth flow when both fail.

Write actions (submit_job, cancel_job, rm, mkdir, chmod, ...) require
``$ALCF_IRI_ALLOW_UNSAFE=1`` and raise RuntimeError otherwise.

Usage:
    python -m chemgraph.mcp.alcf_iri_mcp
    python -m chemgraph.mcp.alcf_iri_mcp --transport streamable_http --port 9010
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, Optional

from mcp.server.fastmcp import FastMCP

from chemgraph.tools.alcf_iri_core import CATEGORIES


mcp = FastMCP(
    name="ChemGraph ALCF IRI Tools",
    instructions="""
        You expose ALCF Facility API tools (https://api.alcf.anl.gov)
        for querying and managing HPC resources at Argonne Leadership
        Computing Facility.

        Tool naming: alcf_<category>_<action>. Categories are facility,
        status, account, compute, filesystem, task, auth.

        Guidelines:
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
        - Prefer alcf_compute_list_jobs with historical=true when the
          user asks about completed jobs; the default (false) is the
          live queue only.
    """,
)


_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "bool": bool,
    "dict": dict,
    "list[str]": list,
}


def _build_wrapper(
    category: str,
    action: str,
    kind: str,
    description: str,
    params_schema: dict,
    invoker: Callable,
) -> tuple[Callable, str, str]:
    """Build an MCP-friendly wrapper for one (category, action).

    The wrapper's signature carries the action's declared params so
    FastMCP can derive a JSON schema from it. Required params become
    positional-or-keyword parameters without defaults; optional params
    default to ``None`` and get an ``Optional[...]`` annotation so
    FastMCP marks them non-required.
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

    if not parameters:
        # FastMCP tolerates zero-arg tools; no placeholder required.
        pass

    def wrapper(**kwargs: Any) -> Any:
        # Drop None values so the invoker's `if v is not None` filters
        # behave the same as they do under the LangChain wrappers.
        cleaned = {k: v for k, v in kwargs.items() if v is not None}
        return invoker(**cleaned)

    # Give the tool a signature FastMCP can introspect.
    wrapper.__name__ = tool_name
    wrapper.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters=parameters,
        return_annotation=Any,
    )

    tag = "read" if kind == "read" else "write"
    full_description = f"[{tag}] {description}"
    return wrapper, tool_name, full_description


def _register_all() -> int:
    """Register one FastMCP tool per IRI action. Returns the tool count."""
    count = 0
    for category, actions in CATEGORIES.items():
        for action, (kind, desc, params_schema, invoker) in actions.items():
            fn, name, description = _build_wrapper(
                category, action, kind, desc, params_schema, invoker,
            )
            mcp.add_tool(fn, name=name, description=description)
            count += 1
    return count


_TOOL_COUNT = _register_all()


if __name__ == "__main__":
    from chemgraph.mcp.server_utils import run_mcp_server

    run_mcp_server(mcp, default_port=9010)
