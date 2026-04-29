"""Main chat interface page for ChemGraph."""

import html as html_mod
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from ase.io import read as ase_read

from chemgraph.memory.store import SessionStore
from chemgraph.models.supported_models import supported_argo_models
from chemgraph.utils.config_utils import (
    get_argo_user_from_nested_config,
    get_base_url_for_model_from_nested_config,
    get_model_options_for_nested_config,
)

from ui.agent_manager import initialize_agent, run_async_callable
from ui.config import load_config
from ui.endpoint import check_local_model_endpoint
from ui.file_utils import (
    changed_recently,
    extract_log_dir_from_messages,
    find_latest_xyz_file_in_dir,
    resolve_output_path,
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
    return get_base_url_for_model_from_nested_config(model_name, config)


def _get_model_options(config: Dict[str, Any]) -> list:
    return get_model_options_for_nested_config(config)


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
    thread_id = config["general"]["thread"]

    # Argo models: disable structured output
    if selected_model in supported_argo_models and structured_output:
        structured_output = False
        st.session_state.ui_notice = (
            "Structured output is disabled for Argo models to avoid JSON parsing errors."
        )

    # ----- Header -----
    st.title("\U0001f9ea ChemGraph")
    st.markdown(
        """
    ChemGraph enables you to perform various **computational chemistry** tasks with
    natural-language queries using AI agents.
    """
    )

    # ----- Quick settings sidebar -----
    selected_model, thread_id = _render_quick_settings(
        config, selected_model, thread_id
    )

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
    _render_agent_status(
        selected_model, selected_workflow, thread_id, endpoint_status
    )

    # ----- Auto-initialize agent -----
    _auto_initialize_agent(
        config,
        selected_model,
        selected_workflow,
        structured_output,
        selected_output,
        generate_report,
        selected_base_url,
    )

    # ----- Conversation history -----
    _render_conversation_history(thread_id)

    # ----- Query input -----
    query = _render_query_input(config, selected_model)

    # ----- Submit -----
    _handle_query_submission(
        query, thread_id, endpoint_status, selected_base_url
    )

    # ----- Footer -----
    _render_footer()


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_quick_settings(
    config: dict, selected_model: str, thread_id: int
) -> tuple[str, int]:
    with st.sidebar.expander("\U0001f527 Quick Settings"):
        st.write("Override settings for this session:")

        if st.checkbox("Override Model"):
            model_options = _get_model_options(config)
            selected_model = st.selectbox(
                "Select Model",
                model_options,
                index=(
                    model_options.index(selected_model)
                    if selected_model in model_options
                    else 0
                ),
            )
            quick_custom_model = st.text_input(
                "Custom model ID (optional)",
                value="",
                key="quick_custom_model",
                help="If set, this overrides the selected model for this session.",
            ).strip()
            if quick_custom_model:
                selected_model = quick_custom_model

        if st.checkbox("Override Thread ID"):
            thread_id = st.number_input(
                "Thread ID", min_value=1, max_value=1000, value=thread_id
            )

        st.info("\U0001f4a1 To make permanent changes, use the Configuration page.")

    return selected_model, thread_id


def _start_new_chat() -> None:
    """Reset conversation state for a fresh chat session."""
    st.session_state.conversation_history.clear()
    st.session_state.current_session_id = None
    st.session_state.session_created = False
    st.session_state.query_input = ""
    st.session_state.last_run_error = None
    st.session_state.last_run_result = None
    st.session_state.last_run_query = None


def _render_session_sidebar() -> None:
    """Render the session management panel in the sidebar."""
    store: Optional[SessionStore] = st.session_state.get("session_store")
    if store is None:
        return

    with st.sidebar.expander("\U0001f4c2 Sessions", expanded=False):
        # New Chat button
        if st.button("\u2795 New Chat", key="new_chat_btn", use_container_width=True):
            _start_new_chat()
            st.rerun()

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
                        logger.warning("Failed to delete session %s: %s", s.session_id, exc)
                    st.rerun()


def _load_session(session_id: str) -> None:
    """Load a stored session into the active conversation."""
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


def _save_exchange_to_store(query: str, result: Any) -> None:
    """Persist a single query/result exchange to the SessionStore.

    Creates the session DB row on the first call, then appends messages.
    """
    store: Optional[SessionStore] = st.session_state.get("session_store")
    if store is None:
        return

    config = st.session_state.config
    model = config["general"]["model"]
    workflow = normalize_workflow_name(config["general"]["workflow"])

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
            )
            st.session_state.session_created = True

        # Build SessionMessage objects for this exchange
        entry = {"query": query, "result": result}
        messages = conversation_entry_to_messages(entry)
        if messages:
            store.save_messages(
                st.session_state.current_session_id, messages
            )
    except Exception as exc:
        # Best-effort persistence -- don't break the UI.
        logger.warning("Failed to save exchange to session store: %s", exc)


def _render_agent_status(
    selected_model: str,
    selected_workflow: str,
    thread_id: int,
    endpoint_status: dict,
) -> None:
    st.sidebar.header("\U0001f171\U0001f172 Agent Status")

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
        if st.session_state.last_run_error:
            st.sidebar.error("Last run error (see verbose info).")

        if st.sidebar.button("\U0001f504 Refresh Agents"):
            st.session_state.agent = None
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
    selected_base_url: Optional[str],
) -> None:
    current_config = (
        selected_model,
        selected_workflow,
        structured_output,
        selected_output,
        generate_report,
        config["general"]["recursion_limit"],
        selected_base_url,
        get_argo_user_from_nested_config(config),
    )

    if (
        st.session_state.agent is None
        or st.session_state.last_config != current_config
    ):
        with st.spinner("\U0001f680 Initializing ChemGraph agents..."):
            st.session_state.agent = initialize_agent(
                selected_model,
                selected_workflow,
                structured_output,
                selected_output,
                generate_report,
                config["general"]["recursion_limit"],
                selected_base_url,
                get_argo_user_from_nested_config(config),
            )
            st.session_state.last_config = current_config


def _render_conversation_history(thread_id: int) -> None:
    if not st.session_state.conversation_history:
        return

    st.subheader("\U0001f5e8\ufe0f Conversation History")

    for idx, entry in enumerate(st.session_state.conversation_history, 1):
        _render_single_exchange(idx, entry, thread_id)
        st.markdown("---")


def _render_single_exchange(idx: int, entry: dict, thread_id: int) -> None:
    """Render one user-query / agent-response exchange."""
    # User bubble
    st.markdown(
        f"""
<div style="background:#e3f2fd;padding:15px;border-radius:15px;margin:10px 0 0 50px;border:1px solid #2196f3;color:#000000;">
  <b style="color:#1976d2;">\U0001f464 You:</b><br><span style="color:#333333;">{html_mod.escape(entry["query"])}</span>
</div>""",
        unsafe_allow_html=True,
    )

    messages = extract_messages_from_result(entry["result"])

    # Find final AI response
    final_answer = _extract_final_answer(messages)

    # Display the AI response
    if final_answer:
        st.markdown(
            f"""
<div style="background:#f1f8e9;padding:15px;border-radius:15px;margin:10px 50px 0 0;border:1px solid #4caf50;color:#000000;">
  <b style="color:#388e3c;">\U0001f171\U0001f172 ChemGraph:</b><br><span style="color:#333333;">{html_mod.escape(final_answer).replace(chr(10), "<br>")}</span>
</div>""",
            unsafe_allow_html=True,
        )

    # Structure visualisation
    html_filename = find_html_filename(messages)
    _render_structure_section(idx, messages, final_answer, entry, html_filename)

    # HTML report
    if html_filename:
        _render_html_report(html_filename, messages)

    # IR spectrum
    if is_infrared_requested(messages):
        _render_ir_spectrum(idx)

    # Debug expander
    _render_verbose_info(idx, messages, entry)


def _extract_final_answer(messages: list) -> str:
    """Walk messages in reverse to find the last non-JSON AI message."""
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
            content = normalize_message_content(
                getattr(message, "content", "")
            ).strip()
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
                log_dir = extract_log_dir_from_messages(messages)
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


def _render_html_report(html_filename: str, messages: list) -> None:
    with st.expander("\U0001f4ca Report", expanded=False):
        try:
            resolved_html = resolve_output_path(html_filename)
            with open(resolved_html, "r", encoding="utf-8") as f:
                html_content = f.read()

            report_structure = extract_xyz_from_report_html(html_content)
            if report_structure:
                display_molecular_structure(
                    report_structure["atomic_numbers"],
                    report_structure["positions"],
                    title="Molecular Structure",
                )

            cleaned_html = strip_viewer_from_report_html(html_content)
            st.components.v1.html(cleaned_html, height=600, scrolling=True)
        except FileNotFoundError:
            st.warning(f"HTML file '{html_filename}' not found")
        except Exception as e:
            st.error(f"Error displaying HTML: {e}")


def _render_ir_spectrum(idx: int) -> None:
    if changed_recently():
        with st.expander("\U0001f50d IR Spectrum", expanded=True):
            col1, col2 = st.columns(2, border=True)

            with col1:
                ir_path = resolve_output_path("ir_spectrum.png")
                if os.path.exists(ir_path):
                    st.image(ir_path)
                else:
                    st.warning("IR spectrum plot not found.")

            with col2:
                freq_path = resolve_output_path("frequencies.csv")
                if not os.path.exists(freq_path):
                    st.warning("Frequencies file not found.")
                else:
                    df = pd.read_csv(
                        freq_path,
                        index_col=False,
                        names=["filename", "frequency"],
                    ).iloc[6:]

                    if not df.empty:
                        st.write("**Select a frequency to visualize:**")
                        freq_options = {
                            f"{float(row['frequency'].strip('i')):.2f} cm\u207b\u00b9": i
                            for i, row in df.iterrows()
                        }
                        selected_freq = st.selectbox(
                            "Frequency",
                            list(freq_options.keys()),
                            index=0,
                            key=f"ir_frequency_select_{idx}",
                        )
                        traj_file = df.loc[freq_options[selected_freq]]["filename"]
                        traj_path = resolve_output_path(traj_file)
                        if not os.path.exists(traj_path):
                            st.warning(
                                f"Trajectory file '{traj_file}' not found."
                            )
                        elif not STMOL_AVAILABLE:
                            st.info(
                                "3D viewer not available; install stmol to animate trajectories."
                            )
                        else:
                            import stmol
                            from ase.io.trajectory import Trajectory

                            traj = Trajectory(traj_path)
                            view = visualize_trajectory(traj)
                            view.zoomTo()
                            stmol.showmol(view, height=400, width=500)
                    else:
                        st.warning("No vibrational frequencies found.")
    else:
        st.warning("IR spectrum not found.")


def _render_verbose_info(idx: int, messages: list, entry: dict) -> None:
    structure = find_structure_in_messages(messages)
    with st.expander(f"\U0001f50d Verbose Info (Query {idx})", expanded=False):
        st.write(f"**Number of messages:** {len(messages)}")
        st.write(f"**Structure found:** {'Yes' if structure else 'No'}")
        if st.session_state.last_run_query == entry.get("query"):
            if st.session_state.last_run_error:
                st.write("**Last run error:**")
                st.code(str(st.session_state.last_run_error))
            if st.session_state.last_run_result is not None:
                st.write("**Raw result (repr):**")
                st.code(repr(st.session_state.last_run_result))

        for i, msg in enumerate(messages):
            if hasattr(msg, "type"):
                msg_type = msg.type
                content = normalize_message_content(msg.content)
            elif isinstance(msg, dict):
                msg_type = msg.get("type", "unknown")
                content = normalize_message_content(msg.get("content", ""))
            else:
                msg_type = type(msg).__name__
                content = normalize_message_content(
                    getattr(msg, "content", str(msg))
                )
            content_preview = (
                (content[:100] + "...") if len(content) > 100 else content
            )
            st.write(f"  **Message {i+1}:** `{msg_type}` - {content_preview}")


def _render_query_input(config: dict, selected_model: str) -> str:
    with st.expander("\U0001f4a1 Example Queries"):
        st.markdown("**Based on your current configuration:**")
        st.markdown(f"- Model: {selected_model}")
        st.markdown(
            f"- Default Calculator: {config['chemistry']['calculators']['default']}"
        )
        st.markdown("- Temperature: 0.0 (optimized for tool calling)")

        examples = [
            "What is the SMILES string for caffeine?",
            f"Optimize the geometry of water molecule using {config['chemistry']['calculators']['default']}",
            "Calculate the infrared spectrum of methanol with xtb calculator",
            "What is the reaction enthalpy of methane combustion using mace_mp",
        ]
        for ex in examples:
            if st.button(ex, key=f"ex_{ex}"):
                st.session_state.query_input = ex
                st.rerun()

    if "query_input" not in st.session_state:
        st.session_state.query_input = ""

    query = st.text_area(
        "Enter your computational chemistry query:",
        value=st.session_state.query_input,
        height=100,
        key="query_text_area",
    )

    if query != st.session_state.query_input:
        st.session_state.query_input = query

    col_send, col_clear, col_refresh = st.columns([2, 1, 1])

    st.session_state._send_clicked = col_send.button(
        "\U0001f680 Send", type="primary", use_container_width=True
    )
    if col_clear.button("\U0001f5d1\ufe0f Clear Chat", use_container_width=True):
        _start_new_chat()
        st.rerun()
    if col_refresh.button("\U0001f504 Refresh", use_container_width=True):
        st.rerun()

    return query


def _handle_query_submission(
    query: str,
    thread_id: int,
    endpoint_status: dict,
    selected_base_url: Optional[str],
) -> None:
    if not st.session_state.get("_send_clicked", False):
        return

    if not endpoint_status["ok"]:
        msg = (
            f"Cannot reach local model endpoint `{selected_base_url}`. "
            f"{endpoint_status['message']}"
        )
        st.session_state.last_run_error = RuntimeError(msg)
        st.error(msg)
    elif not st.session_state.agent:
        st.error("\u274c Agent not ready. Please check configuration and try again.")
        if st.button("\U0001f504 Try Again"):
            st.rerun()
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("ChemGraph agents working...", show_time=True):
            try:
                cfg = {"configurable": {"thread_id": thread_id}}
                st.session_state.last_run_query = query.strip()
                st.session_state.last_run_error = None
                st.session_state.last_run_result = None
                # Capture references eagerly so the lambda never touches
                # st.session_state from the background thread (thread safety).
                agent = st.session_state.agent
                trimmed_query = query.strip()
                result = run_async_callable(
                    lambda: agent.run(trimmed_query, config=cfg)
                )
                st.session_state.last_run_result = result
                st.session_state.conversation_history.append(
                    {
                        "query": query.strip(),
                        "result": result,
                        "thread_id": thread_id,
                    }
                )
                # Persist the exchange to the session store
                _save_exchange_to_store(query.strip(), result)

                st.session_state.query_input = ""
                st.success("\u2705 Done!")
                st.rerun()
            except Exception as exc:
                st.session_state.last_run_error = exc
                st.error(f"Processing error: {exc}")


def _render_footer() -> None:
    st.markdown("---")
    st.markdown(
        """
    ### Quick Help

    **Main Features:** Molecular optimization, vibrational frequencies, SMILES \u2194 structure conversions, 3D visualization

    \U0001f4d6 For detailed information, documentation, and links to research papers, visit the **About ChemGraph** page.
    """
    )

    if st.session_state.ui_notice:
        st.info(st.session_state.ui_notice)
