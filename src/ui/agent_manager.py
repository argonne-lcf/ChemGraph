"""Agent lifecycle management for the ChemGraph Streamlit UI."""

from typing import Optional

import streamlit as st


def initialize_agent(
    model_name: str,
    workflow_type: str,
    structured_output: bool,
    return_option: str,
    generate_report: bool,
    human_supervised: bool,
    recursion_limit: int,
    base_url: Optional[str],
    argo_user: Optional[str],
    log_dir: Optional[str] = None,
):
    """Create a :class:`ChemGraph` agent instance.

    No ``@st.cache_resource`` -- the caller (``_auto_initialize_agent``
    in ``main_interface.py``) already manages caching via
    ``st.session_state.agent`` and ``st.session_state.last_config``.
    Using the decorator caused failed initialisations (``None``) to be
    permanently cached with no way to retry.

    Parameters
    ----------
    model_name : str
        LLM model identifier.
    workflow_type : str
        ChemGraph workflow name.
    structured_output : bool
        Whether structured final output is requested.
    return_option : str
        Agent return mode.
    generate_report : bool
        Whether report generation is enabled.
    human_supervised : bool
        Whether human-supervision tools are enabled.
    recursion_limit : int
        LangGraph recursion limit.
    base_url : str, optional
        Custom model endpoint URL.
    argo_user : str, optional
        Argo username for Argo-hosted models.
    log_dir : str, optional
        Directory for ChemGraph run logs.

    Returns
    -------
    ChemGraph or None
        Initialized agent, or ``None`` if initialization fails.
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
            human_supervised=human_supervised,
            log_dir=log_dir,
        )
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return None


def run_async_callable(fn):
    """Run an async callable and return its result in a sync context.

    Parameters
    ----------
    fn : Callable
        Zero-argument callable returning an awaitable.

    Returns
    -------
    Any
        Result produced by the awaited callable.
    """
    from chemgraph.utils.async_utils import run_async_callable as _impl

    return _impl(fn)
