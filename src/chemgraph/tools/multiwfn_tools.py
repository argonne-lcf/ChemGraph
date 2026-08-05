"""LangChain tool wrapper for scripted Multiwfn analyses."""

from __future__ import annotations

from langchain_core.tools import tool

from chemgraph.schemas.multiwfn_schema import MultiwfnInputSchema, MultiwfnResult
from chemgraph.tools.multiwfn_core import run_multiwfn_core

__all__ = ["run_multiwfn", "run_multiwfn_core"]


@tool
def run_multiwfn(multiwfn_input: MultiwfnInputSchema) -> MultiwfnResult:
    """Run a Multiwfn analysis from an exact sequence of menu responses.

    Use this tool only when the required Multiwfn menu responses are known.
    ``menu_inputs`` must contain one response per prompt, including blank strings
    for Enter and the responses needed to exit cleanly. The Multiwfn executable is
    configured by the user through ``MULTIWFN_EXE`` and cannot be selected by the
    agent.
    """
    return run_multiwfn_core(multiwfn_input)
