"""Agent lifecycle management for the ChemGraph Streamlit UI."""

from pathlib import Path
from typing import Optional, Sequence

import streamlit as st


def build_deepagent_options(
    workflow_type: str,
    workspace: Optional[str],
    skill_dirs: Optional[Sequence[str]],
    discover_skills: bool,
    tool_names: Optional[Sequence[str]],
) -> dict:
    """Translate UI Deep Agent settings into ``ChemGraph`` keyword arguments.

    Mirrors the CLI: the host-shell backend is rooted at the workspace,
    extra skill directories are anchored to absolute paths without checking
    filesystem access, and an explicit ``tools`` list restricts the
    on-demand registry catalog (an empty list disables discovery).

    Parameters
    ----------
    workflow_type : str
        Selected workflow; options are only produced for ``deep_agent``.
    workspace : str, optional
        Workspace directory (empty selects the current directory).
    skill_dirs : sequence of str, optional
        Extra host skill directories.
    discover_skills : bool
        Whether personal/project skills are discovered.
    tool_names : sequence of str, optional
        Registry tool names to expose; ``None`` keeps the full catalog.

    Returns
    -------
    dict
        Keyword arguments for :class:`chemgraph.agent.llm_agent.ChemGraph`.

    Raises
    ------
    ValueError
        When the workspace is not a directory or a tool name is unknown.
    """
    if workflow_type != "deep_agent":
        return {}
    from chemgraph.agent.deepagent_backend import create_host_shell_backend
    from chemgraph.graphs.deep_agent import normalize_skill_sources

    options: dict = {
        "deepagent_backend": create_host_shell_backend(workspace or None),
        "deepagent_discover_skills": bool(discover_skills),
    }
    anchored = tuple(
        str(Path(source).expanduser().absolute())
        for source in normalize_skill_sources(skill_dirs)
    )
    if anchored:
        options["deepagent_skill_dirs"] = anchored
    if tool_names is not None:
        from chemgraph.registry.tools import RegistryError, ToolRegistry

        catalog = ToolRegistry()
        try:
            options["deepagent_tool_registry"] = ToolRegistry(
                catalog.get_spec(name) for name in dict.fromkeys(tool_names)
            )
        except RegistryError as exc:
            raise ValueError(f"Invalid Deep Agent tools: {exc}") from exc
    return options


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
    deepagent_workspace: Optional[str] = None,
    deepagent_skill_dirs: Optional[Sequence[str]] = None,
    deepagent_discover_skills: bool = True,
    deepagent_tool_names: Optional[Sequence[str]] = None,
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
    deepagent_workspace : str, optional
        Deep Agent workspace directory (``deep_agent`` workflow only).
    deepagent_skill_dirs : sequence of str, optional
        Extra Deep Agent skill directories.
    deepagent_discover_skills : bool, optional
        Whether the Deep Agent discovers personal/project skills.
    deepagent_tool_names : sequence of str, optional
        Restrict the Deep Agent's on-demand tool catalog to these names.

    Returns
    -------
    ChemGraph or None
        Initialized agent, or ``None`` if initialization fails.
    """
    try:
        from chemgraph.agent.llm_agent import ChemGraph

        deepagent_options = build_deepagent_options(
            workflow_type,
            deepagent_workspace,
            deepagent_skill_dirs,
            deepagent_discover_skills,
            deepagent_tool_names,
        )
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
            **deepagent_options,
        )
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return None


#: Attributes that identify a conversation and its persistence bookkeeping.
_CONVERSATION_ATTRIBUTES = (
    "uuid",
    "session_store",
    "_session_created",
    "_saved_message_keys",
    "_session_title",
)


def transfer_conversation_state(source, target) -> None:
    """Move an active conversation from one agent instance to another.

    Rebuilding the agent (for example after a provider API key is replaced)
    compiles a fresh graph with an empty in-memory checkpointer, which would
    drop the LangGraph thread state, any pending human-input interrupt, and
    the session bookkeeping that prevents duplicate history writes. Reuse the
    previous checkpointer and session identity on the new instance so the
    conversation continues under the new credentials.

    Parameters
    ----------
    source : ChemGraph
        Agent that currently owns the conversation.
    target : ChemGraph
        Newly constructed agent with the same workflow configuration.
    """
    if source is None or target is None or source is target:
        return
    for name in _CONVERSATION_ATTRIBUTES:
        if hasattr(source, name):
            setattr(target, name, getattr(source, name))
    source_workflow = getattr(source, "workflow", None)
    target_workflow = getattr(target, "workflow", None)
    checkpointer = getattr(source_workflow, "checkpointer", None)
    if checkpointer is not None and target_workflow is not None:
        target_workflow.checkpointer = checkpointer
    if getattr(source, "checkpointer", None) is not None:
        target.checkpointer = source.checkpointer


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
