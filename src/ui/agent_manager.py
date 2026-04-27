"""Agent lifecycle management for the ChemGraph Streamlit UI."""

from typing import Optional

import streamlit as st


def initialize_agent(
    model_name: str,
    workflow_type: str,
    structured_output: bool,
    return_option: str,
    generate_report: bool,
    recursion_limit: int,
    base_url: Optional[str],
    argo_user: Optional[str],
):
    """Create a :class:`ChemGraph` agent instance.

    No ``@st.cache_resource`` -- the caller (``_auto_initialize_agent``
    in ``main_interface.py``) already manages caching via
    ``st.session_state.agent`` and ``st.session_state.last_config``.
    Using the decorator caused failed initialisations (``None``) to be
    permanently cached with no way to retry.
    """
    try:
        from chemgraph.agent.llm_agent import ChemGraph

        return ChemGraph(
            model_name=model_name,
            workflow_type=workflow_type,
            base_url=base_url,
            argo_user=argo_user,
            structured_output=structured_output,
            generate_report=generate_report,
            return_option=return_option,
            recursion_limit=recursion_limit,
        )
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return None


def run_async_callable(fn):
    """Run an async callable and return its result in a sync context."""
    from chemgraph.utils.async_utils import run_async_callable as _impl

    return _impl(fn)
