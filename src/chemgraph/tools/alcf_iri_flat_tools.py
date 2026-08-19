"""Flat @tool variant of the ALCF IRI Facility API integration.

Comparison baseline for :mod:`chemgraph.tools.alcf_iri_tools`. Instead
of exposing 7 category-dispatcher tools (each with a discovery
protocol), this module generates one @tool per action -- ~43 tools
bound to the LLM upfront.

Purpose: benchmark the category-plus-discovery design (current) vs.
flat-all-schemas-upfront (this module). Answers "is the discovery
overhead paying for itself in reduced schema token cost?"

Same underlying implementation via :mod:`chemgraph.tools.alcf_iri_core`;
only the LLM-facing binding shape differs.

Tool naming: `alcf_<category>_<action>`. Names are globally unique
because action names may collide across categories (e.g. `get` lives
in both `facility` and `task`; `list_resources` vs `list_projects`
etc.). The prefix keeps the schemas unambiguous to the LLM.

Not shipped by the single_agent_iri workflow -- use single_agent_iri_flat.
"""

from __future__ import annotations

from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from chemgraph.tools.alcf_iri_core import CATEGORIES


_TYPE_MAP = {
    "str": str,
    "int": int,
    "bool": bool,
    "dict": dict,
    "list[str]": list,
}


def _build_args_model(category: str, action: str, params_schema: dict):
    """Turn an action's params_schema (as stored in CATEGORIES) into
    a Pydantic model LangChain can use to derive the tool's JSON schema.
    """
    fields: dict[str, Any] = {}
    for name, (type_str, required, description) in params_schema.items():
        py_type = _TYPE_MAP.get(type_str, str)
        default = ... if required else None
        fields[name] = (py_type, Field(default, description=description))
    if not fields:
        # StructuredTool requires SOME schema even for zero-arg actions.
        # Add a single ignored placeholder so the LLM has a valid (empty)
        # payload to send. Pydantic rejects leading-underscore field names,
        # so we use "noop" and drop it in the wrapper.
        fields["noop"] = (str, Field(None, description="unused; no args"))
    return create_model(f"AlcfIri_{category}_{action}_Args", **fields)


def _make_tool(category: str, action: str, kind: str, description: str,
               params_schema: dict, invoker: Callable) -> StructuredTool:
    tool_name = f"alcf_{category}_{action}"
    # Wrap invoker so it takes kwargs from the LLM's JSON payload and
    # forwards them to the underlying lambda (which expects **kwargs).
    # Drop the "_" placeholder we may have added.
    def _call(**kwargs) -> Any:
        kwargs.pop("noop", None)
        return invoker(**kwargs)
    args_model = _build_args_model(category, action, params_schema)
    return StructuredTool.from_function(
        func=_call,
        name=tool_name,
        description=f"[{kind}] {description}",
        args_schema=args_model,
    )


def _build_all_tools() -> list[StructuredTool]:
    tools: list[StructuredTool] = []
    for category, actions in CATEGORIES.items():
        for action, (kind, desc, params_schema, invoker) in actions.items():
            tools.append(_make_tool(category, action, kind, desc,
                                    params_schema, invoker))
    return tools


ALCF_IRI_FLAT_TOOLS = _build_all_tools()


__all__ = ["ALCF_IRI_FLAT_TOOLS"]
