"""Main chat interface page for ChemGraph."""

import asyncio
import logging
import os
import pprint
import queue
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from ase.io import read as ase_read

from chemgraph.agent.llm_agent import HumanInputRequired
from chemgraph.memory.store import SessionStore
from chemgraph.models.supported_models import supported_argo_models
from chemgraph.schemas.ase_input import (
    get_available_calculator_names,
    get_default_calculator_name,
)
from chemgraph.utils.config_utils import (
    get_argo_user_from_nested_config,
    get_base_url_for_model_from_nested_config,
)

from ui.agent_manager import initialize_agent
from ui.branding import LOGO_IMAGES, first_existing_asset
from ui.config import load_config
from ui.endpoint import check_local_model_endpoint
from ui.file_utils import (
    extract_log_dir_from_messages,
    find_latest_xyz_file_in_dir,
)
from ui.message_utils import (
    extract_messages_from_result,
    extract_molecular_structure,
    extract_xyz_from_report_html,
    find_html_filename,
    find_structure_in_messages,
    has_structure_signal,
    is_infrared_requested,
    normalize_message_content,
    split_markdown_latex_blocks,
    strip_viewer_from_report_html,
)
from ui.session_utils import (
    conversation_entry_to_messages,
    generate_session_id,
    session_to_conversation_history,
)
from ui.state import init_session_state
from ui.visualization import (
    STMOL_AVAILABLE,
    display_molecular_structure,
    visualize_trajectory,
)

# Re-use the constants from the configuration page
from ui._pages.configuration import normalize_workflow_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thin wrappers around config utilities
# ---------------------------------------------------------------------------


def _get_base_url_for_model(model_name: str, config: Dict[str, Any]) -> Optional[str]:
    """Resolve the configured base URL for a model.

    Parameters
    ----------
    model_name : str
        Selected model identifier.
    config : dict[str, Any]
        Nested UI configuration.

    Returns
    -------
    str or None
        Provider base URL, or ``None`` when not configured.
    """
    return get_base_url_for_model_from_nested_config(model_name, config)


def _initial_ui_log_root() -> str:
    """Return the root directory for per-chat UI artifacts."""
    env_log_dir = os.environ.get("CHEMGRAPH_LOG_DIR")
    if env_log_dir:
        path = Path(env_log_dir).expanduser()
        if path.name.startswith(("session_", "ui_session_")):
            path = path.parent
        return str(path.resolve())
    return str((Path.cwd() / "cg_logs").resolve())


def _ensure_chat_log_dir() -> str:
    """Create and activate a log directory owned by the current chat."""
    if not st.session_state.get("ui_log_root"):
        st.session_state.ui_log_root = _initial_ui_log_root()

    chat_log_dir = st.session_state.get("current_chat_log_dir")
    if not chat_log_dir:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = str(uuid.uuid4())[:8]
        chat_log_dir = str(
            Path(st.session_state.ui_log_root) / f"ui_session_{timestamp}_{suffix}"
        )
        st.session_state.current_chat_log_dir = chat_log_dir

    os.makedirs(chat_log_dir, exist_ok=True)
    os.environ["CHEMGRAPH_LOG_DIR"] = chat_log_dir
    return chat_log_dir


def _resolve_structured_output_for_model(
    model_name: str, structured_output: bool
) -> tuple[bool, Optional[str]]:
    """Disable structured output for Argo models, including quick overrides.

    Parameters
    ----------
    model_name : str
        Selected model identifier.
    structured_output : bool
        Requested structured-output setting.

    Returns
    -------
    tuple[bool, str | None]
        Effective structured-output setting and optional warning message.
    """
    if model_name in supported_argo_models and structured_output:
        return (
            False,
            "Structured output is disabled for Argo models to avoid JSON parsing errors.",
        )
    return structured_output, None


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Main Interface page."""
    init_session_state()

    config = st.session_state.config
    selected_model = config["general"]["model"]
    selected_workflow = normalize_workflow_name(config["general"]["workflow"])
    selected_output = config["general"]["output"]
    structured_output = config["general"]["structured"]
    generate_report = config["general"]["report"]
    human_supervised = config["general"].get("human_supervised", False)
    thread_id = config["general"]["thread"]

    # ----- Header -----
    logo_image = first_existing_asset(LOGO_IMAGES)
    if logo_image:
        st.image(logo_image, width=320)
    else:
        st.title("\U0001f9ea ChemGraph")

    st.markdown("""
    ChemGraph enables you to perform various **computational chemistry** tasks with
    natural-language queries using AI agents.
    """)

    # ----- Calculator availability sidebar -----
    _render_available_calculators_sidebar()
    _render_chat_controls()

    structured_output, ui_notice = _resolve_structured_output_for_model(
        selected_model, structured_output
    )
    st.session_state.ui_notice = ui_notice
    st.session_state.active_model = selected_model
    st.session_state.active_workflow = selected_workflow
    if ui_notice:
        st.info(ui_notice)

    selected_base_url = _get_base_url_for_model(selected_model, config)
    endpoint_status = check_local_model_endpoint(selected_base_url)

    # ----- Session management sidebar -----
    _render_session_sidebar()

    # Reload config button
    if st.sidebar.button("\U0001f504 Reload Config"):
        st.session_state.config = load_config()
        st.success("\u2705 Configuration reloaded!")
        st.rerun()

    # ----- Agent status sidebar -----
    _render_agent_status(selected_model, selected_workflow, thread_id, endpoint_status)

    # ----- Auto-initialize agent -----
    _auto_initialize_agent(
        config,
        selected_model,
        selected_workflow,
        structured_output,
        selected_output,
        generate_report,
        human_supervised,
        selected_base_url,
    )

    # ----- Conversation history -----
    _render_conversation_history(thread_id)

    # ----- Pending interrupt display -----
    _render_pending_interrupt()

    # ----- Example queries -----
    _render_example_queries(config, selected_model)

    # ----- Chat input (handles both normal queries and interrupt responses) -----
    is_interrupt = st.session_state.pending_human_question is not None
    prompt = st.chat_input(
        (
            "Type your response..."
            if is_interrupt
            else "Ask a computational chemistry question..."
        ),
    )

    # Check for example query submitted via button click
    example_query = st.session_state.pop("_pending_example_query", None)
    if example_query:
        prompt = example_query

    if prompt:
        if is_interrupt:
            _handle_human_response(prompt, thread_id)
        else:
            _handle_query_submission(
                prompt, thread_id, endpoint_status, selected_base_url
            )


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_markdown_with_math(text: str) -> None:
    """Render Markdown text, sending display math blocks through ``st.latex``.

    Parameters
    ----------
    text : str
        Markdown text that may contain display math blocks.
    """
    for block_type, content in split_markdown_latex_blocks(text):
        if block_type == "latex":
            st.latex(_prepare_latex_block(content))
        else:
            st.markdown(content)


def _prepare_latex_block(content: str) -> str:
    """Clean display math for Streamlit's KaTeX renderer.

    Parameters
    ----------
    content : str
        Raw LaTeX block content.

    Returns
    -------
    str
        KaTeX-compatible display math.
    """
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1 or r"\begin{" in content:
        return " ".join(lines)
    return "\\begin{aligned}\n" + (r" \\" + "\n").join(lines) + "\n\\end{aligned}"


def _format_calculator_label(calculator_name: str) -> str:
    """Format calculator class names for display.

    Parameters
    ----------
    calculator_name : str
        Calculator class name or label.

    Returns
    -------
    str
        Human-readable calculator label.
    """
    label = calculator_name.removesuffix("Calc")
    if label == "TBLite":
        return "TBLite (xTB, GFN1-xTB, GFN2-xTB)"
    return label


def _render_available_calculators_sidebar() -> None:
    """Render the calculators detected during ChemGraph initialization."""
    available = get_available_calculator_names()
    default = get_default_calculator_name()

    with st.sidebar.expander("\U0001f9ee Available Calculators", expanded=True):
        st.caption("Detected during ChemGraph initialization.")

        for calculator_name in available:
            label = _format_calculator_label(calculator_name)
            if calculator_name == default:
                st.success(f"{label} (default)")
            else:
                st.markdown(f"- {label}")

        st.caption(
            "The agent uses this list when choosing calculators for ASE simulations."
        )


def _start_new_chat() -> None:
    """Reset conversation state for a fresh chat session."""
    st.session_state.conversation_history.clear()
    st.session_state.current_session_id = None
    st.session_state.session_created = False
    st.session_state.query_input = ""
    st.session_state.last_run_error = None
    st.session_state.last_run_result = None
    st.session_state.last_run_query = None
    st.session_state.pop("_pending_example_query", None)
    st.session_state.agent = None
    st.session_state.last_config = None
    st.session_state.current_chat_log_dir = None
    os.environ.pop("CHEMGRAPH_LOG_DIR", None)
    _clear_interrupt_state()


def _render_chat_controls() -> None:
    """Render chat-level actions that must be available even without memory."""
    if st.sidebar.button("New Chat", key="new_chat_btn", use_container_width=True):
        _start_new_chat()
        st.rerun()


def _render_session_sidebar() -> None:
    """Render the session management panel in the sidebar."""
    store: Optional[SessionStore] = st.session_state.get("session_store")
    if store is None:
        return

    with st.sidebar.expander("\U0001f4c2 Sessions", expanded=False):
        # Show current session info
        current_sid = st.session_state.get("current_session_id")
        if current_sid:
            st.caption(f"Active session: `{current_sid}`")

        # List recent sessions
        try:
            sessions = store.list_sessions(limit=10)
        except Exception:
            sessions = []

        if not sessions:
            st.caption("No saved sessions yet.")
            return

        st.markdown("**Recent sessions:**")
        for s in sessions:
            # Highlight the active session
            is_active = current_sid and s.session_id == current_sid
            prefix = "\u25b6 " if is_active else ""
            label = s.title or "Untitled"
            if len(label) > 35:
                label = label[:32] + "..."

            col_load, col_del = st.columns([4, 1])
            with col_load:
                if st.button(
                    f"{prefix}{label}",
                    key=f"load_session_{s.session_id}",
                    use_container_width=True,
                    help=(
                        f"Model: {s.model_name} | "
                        f"Queries: {s.query_count} | "
                        f"{s.updated_at.strftime('%Y-%m-%d %H:%M')}"
                    ),
                ):
                    _load_session(s.session_id)
                    st.rerun()
            with col_del:
                if st.button(
                    "\U0001f5d1",
                    key=f"del_session_{s.session_id}",
                    help="Delete this session",
                ):
                    try:
                        store.delete_session(s.session_id)
                        # If we just deleted the active session, reset
                        if current_sid == s.session_id:
                            _start_new_chat()
                    except Exception as exc:
                        logger.warning(
                            "Failed to delete session %s: %s", s.session_id, exc
                        )
                    st.rerun()


def _load_session(session_id: str) -> None:
    """Load a stored session into the active conversation.

    Parameters
    ----------
    session_id : str
        Session ID or prefix selected in the sidebar.
    """
    store: Optional[SessionStore] = st.session_state.get("session_store")
    if store is None:
        return

    session = store.get_session(session_id)
    if session is None:
        st.sidebar.error(f"Session '{session_id}' not found.")
        return

    # Rebuild conversation_history from stored messages
    st.session_state.conversation_history = session_to_conversation_history(session)
    st.session_state.current_session_id = session.session_id
    st.session_state.session_created = True
    st.session_state.query_input = ""
    st.session_state.last_run_error = None
    st.session_state.last_run_result = None
    st.session_state.last_run_query = None
    st.session_state.current_chat_log_dir = session.log_dir
    if session.log_dir:
        os.environ["CHEMGRAPH_LOG_DIR"] = session.log_dir


def _active_session_metadata() -> tuple[str, str]:
    """Return model/workflow metadata matching the active UI run."""
    config = st.session_state.config
    model = (
        st.session_state.get("pending_interrupt_model")
        or st.session_state.get("active_model")
        or config["general"]["model"]
    )
    workflow = (
        st.session_state.get("pending_interrupt_workflow")
        or st.session_state.get("active_workflow")
        or config["general"]["workflow"]
    )
    return model, normalize_workflow_name(workflow)


def _save_exchange_to_store(query: str, result: Any) -> None:
    """Persist a single query/result exchange to the SessionStore.

    Creates the session DB row on the first call, then appends messages.

    Parameters
    ----------
    query : str
        User query text.
    result : Any
        Agent result to persist as session messages.
    """
    store: Optional[SessionStore] = st.session_state.get("session_store")
    if store is None:
        return

    model, workflow = _active_session_metadata()

    try:
        # Create the session row on the first exchange
        if not st.session_state.session_created:
            sid = generate_session_id()
            st.session_state.current_session_id = sid
            title = SessionStore.generate_title(query)
            store.create_session(
                session_id=sid,
                model_name=model,
                workflow_type=workflow,
                title=title,
                log_dir=st.session_state.get("current_chat_log_dir"),
            )
            st.session_state.session_created = True

        # Build SessionMessage objects for this exchange
        entry = {"query": query, "result": result}
        messages = conversation_entry_to_messages(entry)
        if messages:
            store.save_messages(st.session_state.current_session_id, messages)
    except Exception as exc:
        # Best-effort persistence -- don't break the UI.
        logger.warning("Failed to save exchange to session store: %s", exc)


def _render_agent_status(
    selected_model: str,
    selected_workflow: str,
    thread_id: int,
    endpoint_status: dict,
) -> None:
    """Render sidebar status for the active agent.

    Parameters
    ----------
    selected_model : str
        Selected model name.
    selected_workflow : str
        Selected workflow name.
    thread_id : int
        Current LangGraph thread ID.
    endpoint_status : dict
        Local endpoint status dictionary.
    """
    st.sidebar.header("Agent Status")

    if st.session_state.agent:
        st.sidebar.success("\u2705 Agents Ready")
        st.sidebar.info(f"\U0001f9e0 Model: {selected_model}")
        st.sidebar.info(f"\u2699\ufe0f Workflow: {selected_workflow}")
        st.sidebar.info(f"\U0001f517 Thread ID: {thread_id}")
        st.sidebar.info(
            f"\U0001f4ac Messages: {len(st.session_state.conversation_history)}"
        )
        if endpoint_status["ok"]:
            st.sidebar.caption(f"LLM endpoint: {endpoint_status['message']}")
        else:
            st.sidebar.error(f"LLM endpoint issue: {endpoint_status['message']}")
        if st.session_state.pending_human_question is not None:
            st.sidebar.warning("Waiting for your input...")
        if st.session_state.last_run_error:
            st.sidebar.error("Last run error (see verbose info).")

        if st.sidebar.button("\U0001f504 Refresh Agents"):
            st.session_state.agent = None
            # Checkpoint is lost on re-init, so clear interrupt state
            _clear_interrupt_state()
            st.rerun()
    else:
        st.sidebar.error("\u274c Agents Not Ready")
        st.sidebar.info("Agents will initialize automatically...")
        if not endpoint_status["ok"]:
            st.sidebar.error(f"LLM endpoint issue: {endpoint_status['message']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**\u2699\ufe0f Configuration**")
    st.sidebar.markdown(
        "Use the Configuration page to modify settings, API endpoints, and chemistry parameters."
    )
    st.sidebar.markdown("Current config loaded from: `config.toml`")


def _auto_initialize_agent(
    config: dict,
    selected_model: str,
    selected_workflow: str,
    structured_output: bool,
    selected_output: str,
    generate_report: bool,
    human_supervised: bool,
    selected_base_url: Optional[str],
) -> None:
    """Initialize or refresh the cached Streamlit agent when config changes.

    Parameters
    ----------
    config : dict
        Nested UI configuration.
    selected_model : str
        Selected model name.
    selected_workflow : str
        Selected workflow name.
    structured_output : bool
        Effective structured-output setting.
    selected_output : str
        Agent return mode.
    generate_report : bool
        Whether report generation is enabled.
    human_supervised : bool
        Whether human-supervision tools are enabled.
    selected_base_url : str, optional
        Model endpoint URL.
    """
    current_config = (
        selected_model,
        selected_workflow,
        structured_output,
        selected_output,
        generate_report,
        human_supervised,
        config["general"]["recursion_limit"],
        selected_base_url,
        get_argo_user_from_nested_config(config),
        st.session_state.get("current_chat_log_dir"),
    )

    if st.session_state.agent is None or st.session_state.last_config != current_config:
        with st.spinner("\U0001f680 Initializing ChemGraph agents..."):
            chat_log_dir = _ensure_chat_log_dir()
            agent = initialize_agent(
                selected_model,
                selected_workflow,
                structured_output,
                selected_output,
                generate_report,
                human_supervised,
                config["general"]["recursion_limit"],
                selected_base_url,
                get_argo_user_from_nested_config(config),
                log_dir=chat_log_dir,
            )
            st.session_state.agent = agent
            if agent is not None:
                st.session_state.last_config = (
                    selected_model,
                    selected_workflow,
                    structured_output,
                    selected_output,
                    generate_report,
                    human_supervised,
                    config["general"]["recursion_limit"],
                    selected_base_url,
                    get_argo_user_from_nested_config(config),
                    chat_log_dir,
                )
            else:
                st.session_state.last_config = None


def _render_conversation_history(thread_id: int) -> None:
    """Render all saved conversation exchanges.

    Parameters
    ----------
    thread_id : int
        Current LangGraph thread ID.
    """
    if not st.session_state.conversation_history:
        return

    for idx, entry in enumerate(st.session_state.conversation_history, 1):
        _render_single_exchange(idx, entry, thread_id)


def _render_single_exchange(idx: int, entry: dict, thread_id: int) -> None:
    """Render one user-query / agent-response exchange.

    Parameters
    ----------
    idx : int
        One-based exchange index.
    entry : dict
        Conversation-history entry.
    thread_id : int
        Current LangGraph thread ID.
    """
    # User message
    with st.chat_message("user"):
        st.markdown(entry["query"])

    # Interrupt exchanges (if any occurred during this query)
    for exch in entry.get("interrupt_exchanges", []):
        with st.chat_message("assistant"):
            _render_markdown_with_math(exch["question"])
        with st.chat_message("user"):
            st.markdown(exch["answer"])

    messages = extract_messages_from_result(entry["result"])

    # Find final AI response
    final_answer = _extract_final_answer(messages)

    # Display the AI response with visualizations
    with st.chat_message("assistant"):
        if final_answer:
            _render_markdown_with_math(final_answer)

        # Structure visualisation
        html_filename = find_html_filename(messages)
        _render_structure_section(idx, messages, final_answer, entry, html_filename)

        # HTML report
        if html_filename:
            _render_html_report(idx, html_filename, messages, entry)

        # IR spectrum
        if is_infrared_requested(messages):
            _render_ir_spectrum(idx, messages, entry)

        # Debug expander
        _render_verbose_info(idx, messages, entry)


def _extract_final_answer(messages: list) -> str:
    """Walk messages in reverse to find the last non-JSON AI message.

    Parameters
    ----------
    messages : list
        Message-like objects or dictionaries.

    Returns
    -------
    str
        Final displayable answer text, or an empty string.
    """
    final_answer = ""
    for message in reversed(messages):
        if hasattr(message, "content") and hasattr(message, "type"):
            content = normalize_message_content(message.content).strip()
            if message.type == "ai" and content:
                if not (
                    content.startswith("{")
                    and content.endswith("}")
                    and "numbers" in content
                ):
                    final_answer = content
                    break
        elif isinstance(message, dict):
            content = normalize_message_content(message.get("content", "")).strip()
            if message.get("type") == "ai" and content:
                if not (
                    content.startswith("{")
                    and content.endswith("}")
                    and "numbers" in content
                ):
                    final_answer = content
                    break
        elif hasattr(message, "content"):
            content = normalize_message_content(getattr(message, "content", "")).strip()
            if content and not (
                content.startswith("{")
                and content.endswith("}")
                and "numbers" in content
            ):
                final_answer = content
                break
    return final_answer


def _render_structure_section(
    idx: int,
    messages: list,
    final_answer: str,
    entry: dict,
    html_filename: Optional[str],
) -> None:
    """Render molecular structure artifacts for an exchange.

    Parameters
    ----------
    idx : int
        One-based exchange index.
    messages : list
        Message-like objects from the exchange.
    final_answer : str
        Final assistant answer text.
    entry : dict
        Conversation-history entry.
    html_filename : str, optional
        HTML report path/filename, if detected.
    """
    structure = find_structure_in_messages(messages)
    if structure:
        display_molecular_structure(
            structure["atomic_numbers"],
            structure["positions"],
            title=f"Molecular Structure (Query {idx})",
        )
    else:
        structure_from_text = extract_molecular_structure(final_answer)
        if structure_from_text:
            display_molecular_structure(
                structure_from_text["atomic_numbers"],
                structure_from_text["positions"],
                title=f"Structure from Response {idx}",
            )
        elif not html_filename:
            if has_structure_signal(messages, entry.get("query", ""), final_answer):
                log_dir = _artifact_log_dir(messages, entry)
                if log_dir and os.path.isdir(log_dir):
                    latest_xyz = find_latest_xyz_file_in_dir(log_dir)
                    if latest_xyz:
                        try:
                            atoms = ase_read(latest_xyz)
                            display_molecular_structure(
                                atoms.get_atomic_numbers().tolist(),
                                atoms.get_positions().tolist(),
                                title=f"Structure from {Path(latest_xyz).name}",
                            )
                        except Exception as exc:
                            st.warning(f"Failed to load XYZ structure: {exc}")


def _render_html_report(
    idx: int, html_filename: str, messages: list, entry: dict
) -> None:
    """Render an HTML report expander and download button.

    Parameters
    ----------
    idx : int
        One-based exchange index.
    html_filename : str
        HTML report path or filename.
    messages : list
        Message-like objects from the exchange.
    entry : dict
        Conversation-history entry.
    """
    with st.expander("\U0001f4ca Report", expanded=False):
        try:
            resolved_html = _resolve_artifact_path(
                html_filename,
                _artifact_log_dir(messages, entry),
            )
            with open(resolved_html, "r", encoding="utf-8") as f:
                html_content = f.read()

            report_structure = extract_xyz_from_report_html(html_content)
            if report_structure:
                display_molecular_structure(
                    report_structure["atomic_numbers"],
                    report_structure["positions"],
                    title=f"Molecular Structure (Report {idx})",
                )

            cleaned_html = strip_viewer_from_report_html(html_content)
            st.download_button(
                "Download HTML Report",
                data=html_content,
                file_name=Path(resolved_html).name,
                mime="text/html",
                key=f"download_report_{idx}",
            )
            st.components.v1.html(cleaned_html, height=600, scrolling=True)
        except FileNotFoundError:
            st.warning(f"HTML file '{html_filename}' not found")
        except Exception as e:
            st.error(f"Error displaying HTML: {e}")


def _artifact_log_dir(messages: list, entry: dict) -> Optional[str]:
    """Return the log directory tied to a specific conversation entry.

    Parameters
    ----------
    messages : list
        Message-like objects from the exchange.
    entry : dict
        Conversation-history entry.

    Returns
    -------
    str or None
        Artifact/log directory, if found.
    """
    entry_log_dir = entry.get("log_dir")
    if entry_log_dir:
        return entry_log_dir
    return extract_log_dir_from_messages(messages)


def _latest_artifact_path(directory: Optional[str], pattern: str) -> Optional[str]:
    """Return the newest shallow match for an output artifact pattern.

    Parameters
    ----------
    directory : str, optional
        Directory to search.
    pattern : str
        Glob pattern to match.

    Returns
    -------
    str or None
        Newest matching file path, or ``None``.
    """
    if not directory or not os.path.isdir(directory):
        return None

    candidates: list[Path] = []
    try:
        candidates.extend(path for path in Path(directory).glob(pattern) if path.is_file())
    except OSError:
        return None

    if not candidates:
        return None
    return str(max(candidates, key=lambda path: path.stat().st_mtime))


def _resolve_artifact_path(filename: str, directory: Optional[str]) -> str:
    """Resolve an artifact path relative to its run directory when known.

    Parameters
    ----------
    filename : str
        Absolute or relative artifact path.
    directory : str, optional
        Run artifact directory.

    Returns
    -------
    str
        Resolved artifact path.
    """
    if os.path.isabs(filename):
        return filename
    if directory:
        return str(Path(directory) / filename)
    return filename


def _is_linear_geometry(atoms) -> bool:
    """Return whether an ASE ``Atoms`` object is (near-)linear.

    A linear molecule has one vanishing principal moment of inertia.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to inspect.

    Returns
    -------
    bool
        ``True`` when the geometry is linear.
    """
    try:
        moments = sorted(abs(float(m)) for m in atoms.get_moments_of_inertia())
    except Exception:
        return False
    if len(moments) < 3 or moments[-1] <= 0:
        return False
    return moments[0] < 1e-3 * moments[-1]


def _num_nonvibrational_modes(total_modes: int, log_dir: Optional[str]) -> int:
    """Return how many leading translational/rotational modes to skip.

    Non-linear molecules have 6 (3 translations + 3 rotations); linear
    molecules have only 5.  Hardcoding 6 silently dropped a genuine
    vibration for linear species (CO2, HCN, diatomics).

    Parameters
    ----------
    total_modes : int
        Total number of rows in the frequency table (``3N``).
    log_dir : str, optional
        Directory to search for a structure file used to test linearity.

    Returns
    -------
    int
        Number of leading modes to discard.
    """
    n_atoms = total_modes // 3
    if n_atoms <= 2:
        # Single atom -> no vibrations; diatomic -> always linear (5 modes).
        return min(total_modes, 5)
    if log_dir:
        xyz = find_latest_xyz_file_in_dir(log_dir)
        if xyz:
            try:
                atoms = ase_read(xyz)
                if _is_linear_geometry(atoms):
                    return 5
            except Exception:
                pass
    return 6


def _render_ir_spectrum(idx: int, messages: list, entry: dict) -> None:
    """Render IR spectrum plot, frequency table, and trajectory viewer.

    Parameters
    ----------
    idx : int
        One-based exchange index.
    messages : list
        Message-like objects from the exchange.
    entry : dict
        Conversation-history entry.
    """
    log_dir = _artifact_log_dir(messages, entry)
    ir_path = _latest_artifact_path(log_dir, "ir_spectrum*.png")
    freq_path = _latest_artifact_path(log_dir, "frequencies*.csv")

    if not ir_path and not freq_path:
        st.warning("IR spectrum not found.")
        return

    with st.expander("\U0001f50d IR Spectrum", expanded=True):
        col1, col2 = st.columns(2, border=True)

        with col1:
            if ir_path and os.path.exists(ir_path):
                st.image(ir_path)
            else:
                st.warning("IR spectrum plot not found.")

        with col2:
            if not freq_path or not os.path.exists(freq_path):
                st.warning("Frequencies file not found.")
                return

            df = pd.read_csv(
                freq_path,
                index_col=False,
                names=["filename", "frequency"],
            )
            n_skip = _num_nonvibrational_modes(len(df), log_dir)
            modes = df.iloc[n_skip:] if len(df) > n_skip else df

            if modes.empty:
                st.warning("No vibrational frequencies found.")
                return

            st.write("**Select a frequency to visualize:**")
            freq_options = {}
            for mode_idx, row in modes.iterrows():
                freq_text = str(row["frequency"]).strip()
                suffix = "i" if freq_text.endswith("i") else ""
                try:
                    freq_value = float(freq_text.rstrip("i"))
                    label = f"Mode {mode_idx}: {freq_value:.2f}{suffix} cm\u207b\u00b9"
                except ValueError:
                    label = f"Mode {mode_idx}: {freq_text} cm\u207b\u00b9"
                freq_options[label] = mode_idx

            selected_freq = st.selectbox(
                "Frequency",
                list(freq_options.keys()),
                index=0,
                key=f"ir_frequency_select_{idx}",
            )
            traj_file = str(modes.loc[freq_options[selected_freq]]["filename"])
            traj_path = _resolve_artifact_path(traj_file, log_dir)
            if not os.path.exists(traj_path):
                st.warning(f"Trajectory file '{traj_file}' not found.")
            elif not STMOL_AVAILABLE:
                st.info("3D viewer not available; install stmol to animate trajectories.")
            else:
                import stmol
                from ase.io.trajectory import Trajectory

                traj = Trajectory(traj_path)
                view = visualize_trajectory(traj)
                view.zoomTo()
                stmol.showmol(view, height=400, width=500)


def _render_verbose_info(idx: int, messages: list, entry: dict) -> None:
    """Render raw result/debug information for an exchange.

    Parameters
    ----------
    idx : int
        One-based exchange index.
    messages : list
        Message-like objects from the exchange.
    entry : dict
        Conversation-history entry.
    """
    structure = find_structure_in_messages(messages)
    with st.expander(f"\U0001f50d Verbose Info (Query {idx})", expanded=False):
        st.write(f"**Number of messages:** {len(messages)}")
        st.write(f"**Structure found:** {'Yes' if structure else 'No'}")
        raw_result = entry.get("result")
        if st.session_state.last_run_query == entry.get("query"):
            if st.session_state.last_run_error:
                st.write("**Last run error:**")
                st.code(str(st.session_state.last_run_error))
            if st.session_state.last_run_result is not None:
                raw_result = st.session_state.last_run_result

        st.write("**Raw result:**")
        st.code(pprint.pformat(raw_result, width=1, compact=False), language="text")


def _render_example_queries(config: dict, selected_model: str) -> None:
    """Show example queries that the user can click to submit directly.

    Parameters
    ----------
    config : dict
        Nested UI configuration.
    selected_model : str
        Selected model name.
    """
    # Hide after the first message or during an interrupt
    if (
        st.session_state.conversation_history
        or st.session_state.pending_human_question is not None
    ):
        return

    with st.expander("Example Queries", expanded=False):
        st.markdown("**Based on your current configuration:**")
        st.markdown(f"- Model: {selected_model}")
        st.markdown(
            f"- Default Calculator: {config['chemistry']['calculators']['default']}"
        )

        examples = [
            "What is the SMILES string for caffeine?",
            f"Optimize the geometry of water molecule using {config['chemistry']['calculators']['default']}",
            "Calculate the infrared spectrum of methanol with xtb calculator",
            "What is the reaction enthalpy of methane combustion using mace_mp",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state._pending_example_query = ex
                st.rerun()


def _render_pending_interrupt() -> None:
    """Show the agent's pending question and any prior interrupt exchanges."""
    question = st.session_state.pending_human_question
    if question is None:
        return

    # Show the original user query that triggered the interrupt
    original_query = st.session_state.pending_interrupt_query
    if original_query:
        with st.chat_message("user"):
            _render_markdown_with_math(original_query)

    # Show any prior interrupt exchanges in this chain
    for exch in st.session_state.interrupt_exchanges:
        with st.chat_message("assistant"):
            _render_markdown_with_math(exch["question"])
        with st.chat_message("user"):
            _render_markdown_with_math(exch["answer"])

    # Show the current pending question
    with st.chat_message("assistant"):
        st.info("The agent needs your input to continue.", icon="\u2753")
        _render_markdown_with_math(question)

    # Cancel button
    if st.button("Cancel", key="cancel_interrupt"):
        _clear_interrupt_state()
        st.rerun()


def _clear_interrupt_state() -> None:
    """Clear all interrupt-related session state."""
    st.session_state.pending_human_question = None
    st.session_state.pending_interrupt_config = None
    st.session_state.pending_interrupt_query = None
    st.session_state.pending_interrupt_thread_id = None
    st.session_state.pending_interrupt_prev_msg_count = 0
    st.session_state.pending_interrupt_model = None
    st.session_state.pending_interrupt_workflow = None
    st.session_state.pending_interrupt_log_dir = None
    st.session_state.interrupt_count = 0
    st.session_state.interrupt_exchanges = []


def _classify_message(msg):
    """Classify a LangGraph message for UI display.

    Parameters
    ----------
    msg : Any
        LangGraph/LangChain message to classify.

    Returns
    -------
    tuple or None
        ``("tool_call", [tool_names])``, ``("tool_result", tool_name)``, or
        ``None`` when not relevant for display.
    """
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        names = [tc.get("name", "unknown") for tc in tool_calls if isinstance(tc, dict)]
        if names:
            return ("tool_call", names)
    if getattr(msg, "type", None) == "tool":
        name = getattr(msg, "name", None)
        if name:
            return ("tool_result", name)
    return None


def _stream_workflow(stream_input, config, agent, msg_queue):
    """Run the agent workflow in a background thread, pushing events to a queue.

    Events pushed:
        ("tool_call", [tool_names])   — agent is calling tool(s)
        ("tool_result", tool_name)    — a tool finished
        ("interrupt", question_str)
        ("done", last_state)
        ("error", exception)

    Parameters
    ----------
    stream_input : dict or Command
        Initial workflow input or resume command.
    config : dict
        LangGraph run configuration.
    agent : ChemGraph
        Active ChemGraph agent.
    msg_queue : queue.Queue
        Queue receiving stream events for the UI thread.
    """
    from langgraph.errors import GraphInterrupt

    async def _run():
        """Stream the workflow and enqueue UI events."""
        prev_msgs: list = []
        last_st = None
        interrupt_val = None

        try:
            async for s in agent.workflow.astream(
                stream_input, stream_mode="values", config=config
            ):
                if "__interrupt__" in s:
                    int_data = s["__interrupt__"]
                    if isinstance(int_data, (list, tuple)) and int_data:
                        interrupt_val = int_data[0].value
                    elif hasattr(int_data, "value"):
                        interrupt_val = int_data.value
                    else:
                        interrupt_val = {"question": "The workflow needs your input."}

                if "messages" in s and s["messages"] != prev_msgs:
                    new_message = s["messages"][-1]
                    classified = _classify_message(new_message)
                    if classified:
                        msg_queue.put(classified)
                    prev_msgs = s["messages"]
                last_st = s
        except GraphInterrupt as gi:
            interrupts = gi.args[0] if gi.args else []
            if interrupts:
                interrupt_val = interrupts[0].value
            else:
                interrupt_val = {"question": "The workflow needs your input."}

        # Check checkpoint for pending interrupts
        if interrupt_val is None:
            try:
                snapshot = agent.workflow.get_state(config)
                if snapshot and snapshot.tasks:
                    for t in snapshot.tasks:
                        t_interrupts = getattr(t, "interrupts", None)
                        if t_interrupts:
                            interrupt_val = t_interrupts[0].value
                            break
            except Exception:
                pass

        if interrupt_val is not None:
            if isinstance(interrupt_val, dict):
                q = interrupt_val.get(
                    "question",
                    interrupt_val.get("message", str(interrupt_val)),
                )
            else:
                q = str(interrupt_val)
            msg_queue.put(("interrupt", q))
        else:
            msg_queue.put(("done", last_st))

    try:
        asyncio.run(_run())
    except HumanInputRequired as hir:
        msg_queue.put(("interrupt", hir.question))
    except Exception as exc:
        msg_queue.put(("error", exc))


def _poll_and_display(msg_queue, status_container, placeholder, thread):
    """Poll the message queue and render a compact tool-call log.

    Uses a single ``st.empty()`` placeholder to re-render the full list
    each time, so completed tools get a checkmark and only the active
    tool shows a spinner indicator.

    Returns:
        ("done", last_state) | ("interrupt", question) | ("error", exception)

    Parameters
    ----------
    msg_queue : queue.Queue
        Queue receiving stream events.
    status_container : DeltaGenerator
        Streamlit status container.
    placeholder : DeltaGenerator
        Placeholder used for the tool-call log.
    thread : threading.Thread
        Background stream thread.

    Returns
    -------
    tuple
        ``("done", state)``, ``("interrupt", question)``, or
        ``("error", exception)``.
    """
    completed: list[str] = []  # tools that finished
    active: list[str] = []  # tools currently running

    def _render():
        """Render the current tool-call status list."""
        lines = []
        for name in completed:
            lines.append(f"- :green[**{name}**] :white_check_mark:")
        for name in active:
            lines.append(f"- **{name}** :hourglass_flowing_sand:")
        placeholder.markdown("\n".join(lines) if lines else "")

    while True:
        try:
            event_type, event_data = msg_queue.get(timeout=0.1)
        except queue.Empty:
            if not thread.is_alive():
                try:
                    event_type, event_data = msg_queue.get_nowait()
                except queue.Empty:
                    return ("error", RuntimeError("Stream ended without result."))
            else:
                continue

        if event_type == "tool_call":
            # Mark previously active tools as completed
            completed.extend(active)
            active.clear()
            active.extend(event_data)
            label = ", ".join(event_data)
            status_container.update(label=f"Running {label}", state="running")
            _render()
        elif event_type == "tool_result":
            # Move this specific tool from active to completed
            if event_data in active:
                active.remove(event_data)
            if event_data not in completed:
                completed.append(event_data)
            if active:
                status_container.update(
                    label=f"Running {', '.join(active)}", state="running"
                )
            else:
                status_container.update(label="Thinking...", state="running")
            _render()
        elif event_type in ("done", "interrupt", "error"):
            # Final render — mark everything as completed
            completed.extend(active)
            active.clear()
            _render()
            return (event_type, event_data)


def _handle_query_submission(
    query: str,
    thread_id: int,
    endpoint_status: dict,
    selected_base_url: Optional[str],
) -> None:
    """Handle a submitted user query and stream the workflow response.

    Parameters
    ----------
    query : str
        User query text.
    thread_id : int
        Current LangGraph thread ID.
    endpoint_status : dict
        Local endpoint status dictionary.
    selected_base_url : str, optional
        Model endpoint URL used in error messages.
    """
    if not endpoint_status["ok"]:
        msg = (
            f"Cannot reach local model endpoint `{selected_base_url}`. "
            f"{endpoint_status['message']}"
        )
        st.session_state.last_run_error = RuntimeError(msg)
        st.error(msg)
        return
    if not st.session_state.agent:
        st.error("Agent not ready. Please check configuration and try again.")
        return
    if not query.strip():
        return

    trimmed_query = query.strip()
    agent = st.session_state.agent
    cfg = {"configurable": {"thread_id": str(thread_id)}}
    cfg["recursion_limit"] = agent.recursion_limit
    st.session_state.last_run_query = trimmed_query
    st.session_state.last_run_error = None
    st.session_state.last_run_result = None

    # Agent setup (mirroring agent.run() preamble)
    if agent.log_dir:
        os.environ["CHEMGRAPH_LOG_DIR"] = agent.log_dir
    try:
        agent._ensure_session(trimmed_query)
    except Exception:
        pass

    # Snapshot message count before streaming so we can isolate new messages
    prev_msg_count = 0
    try:
        snapshot = agent.workflow.get_state(cfg)
        if snapshot and snapshot.values:
            prev_msg_count = len(snapshot.values.get("messages", []))
    except Exception:
        pass

    # Show the user's message immediately
    with st.chat_message("user"):
        st.markdown(trimmed_query)

    # Stream agent response with live tool-call display
    with st.chat_message("assistant"):
        msg_q: queue.Queue = queue.Queue()
        inputs = {"messages": trimmed_query}

        stream_thread = threading.Thread(
            target=_stream_workflow,
            args=(inputs, cfg, agent, msg_q),
            daemon=True,
        )

        status = st.status("Thinking...", expanded=True)
        with status:
            tool_log = st.empty()
        stream_thread.start()
        event_type, event_data = _poll_and_display(
            msg_q, status, tool_log, stream_thread
        )
        stream_thread.join(timeout=5)

        if event_type == "done":
            status.update(label="Complete", state="complete", expanded=False)
            last_state = event_data
            if last_state is None:
                st.error("Workflow produced no output.")
                return

            # Only keep messages from this query (not prior thread history)
            all_msgs = last_state.get("messages", [])
            new_msgs = all_msgs[prev_msg_count:]
            result = {"messages": new_msgs}

            # Save messages to persistent session store (best-effort)
            try:
                agent._save_messages_to_store(last_state, trimmed_query)
            except Exception:
                pass

            st.session_state.last_run_result = result
            st.session_state.conversation_history.append(
                {
                    "query": trimmed_query,
                    "result": result,
                    "thread_id": thread_id,
                    "log_dir": agent.log_dir,
                }
            )
            _save_exchange_to_store(trimmed_query, result)
            st.session_state.query_input = ""
            st.rerun()

        elif event_type == "interrupt":
            status.update(label="Waiting for input", state="complete", expanded=False)
            cfg_for_resume = dict(cfg)
            st.session_state.pending_human_question = event_data
            st.session_state.pending_interrupt_config = cfg_for_resume
            st.session_state.pending_interrupt_query = trimmed_query
            st.session_state.pending_interrupt_thread_id = thread_id
            st.session_state.pending_interrupt_prev_msg_count = prev_msg_count
            st.session_state.pending_interrupt_model = st.session_state.get(
                "active_model"
            )
            st.session_state.pending_interrupt_workflow = st.session_state.get(
                "active_workflow"
            )
            st.session_state.pending_interrupt_log_dir = agent.log_dir
            st.session_state.interrupt_count = 1
            st.session_state.interrupt_exchanges = []
            st.rerun()

        else:  # error
            status.update(label="Error", state="error", expanded=False)
            st.session_state.last_run_error = event_data
            st.error(f"Processing error: {event_data}")


def _handle_human_response(answer: str, thread_id: int) -> None:
    """Resume the agent workflow with the human's answer.

    Parameters
    ----------
    answer : str
        Human response to the pending interrupt question.
    thread_id : int
        Current LangGraph thread ID.
    """
    from langgraph.types import Command

    agent = st.session_state.agent
    resume_config = st.session_state.pending_interrupt_config
    original_query = st.session_state.pending_interrupt_query
    current_question = st.session_state.pending_human_question
    interrupt_count = st.session_state.interrupt_count

    if agent is None or resume_config is None:
        st.error("Agent was re-initialized. Please submit your query again.")
        _clear_interrupt_state()
        return
    if agent.log_dir:
        os.environ["CHEMGRAPH_LOG_DIR"] = agent.log_dir

    MAX_INTERRUPTS = 10

    # Record this exchange
    st.session_state.interrupt_exchanges.append(
        {"question": current_question, "answer": answer}
    )

    # Show the user's reply immediately
    with st.chat_message("user"):
        st.markdown(answer)

    # Stream resumed agent response
    with st.chat_message("assistant"):
        msg_q: queue.Queue = queue.Queue()
        resume_cmd = Command(resume=answer)

        stream_thread = threading.Thread(
            target=_stream_workflow,
            args=(resume_cmd, resume_config, agent, msg_q),
            daemon=True,
        )

        status = st.status("Processing your response...", expanded=True)
        with status:
            tool_log = st.empty()
        stream_thread.start()
        event_type, event_data = _poll_and_display(
            msg_q, status, tool_log, stream_thread
        )
        stream_thread.join(timeout=5)

        if event_type == "done":
            status.update(label="Complete", state="complete", expanded=False)
            result_state = event_data

            if result_state is None:
                st.error("Resume produced no output.")
                _clear_interrupt_state()
                return

            # Only keep messages from this query (not prior thread history)
            prev_msg_count = st.session_state.get("pending_interrupt_prev_msg_count", 0)
            all_msgs = result_state.get("messages", [])
            new_msgs = all_msgs[prev_msg_count:]
            final_result = {"messages": new_msgs}

            exchanges = list(st.session_state.interrupt_exchanges)
            st.session_state.last_run_result = final_result
            st.session_state.conversation_history.append(
                {
                    "query": original_query,
                    "result": final_result,
                    "thread_id": thread_id,
                    "log_dir": st.session_state.get("pending_interrupt_log_dir")
                    or agent.log_dir,
                    "interrupt_exchanges": exchanges,
                }
            )
            _save_exchange_to_store(original_query, final_result)
            st.session_state.query_input = ""
            _clear_interrupt_state()
            st.rerun()

        elif event_type == "interrupt":
            status.update(label="Waiting for input", state="complete", expanded=False)
            new_count = interrupt_count + 1
            if new_count > MAX_INTERRUPTS:
                st.error(
                    "Agent exceeded maximum number of follow-up questions. Aborting."
                )
                _clear_interrupt_state()
                return
            st.session_state.pending_human_question = event_data
            st.session_state.interrupt_count = new_count
            st.rerun()

        else:  # error
            status.update(label="Error", state="error", expanded=False)
            st.session_state.last_run_error = event_data
            st.error(f"Error during resume: {event_data}")
            _clear_interrupt_state()
