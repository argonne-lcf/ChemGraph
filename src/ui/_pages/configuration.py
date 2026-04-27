"""Configuration editor page."""

import copy
import os
from typing import Any, Dict

import streamlit as st
import toml

from ui.config import get_default_config, load_config, save_config

# ---------------------------------------------------------------------------
# Constants shared with the main app
# ---------------------------------------------------------------------------

WORKFLOW_ALIASES: Dict[str, str] = {
    "python_repl": "python_relp",
    "graspa_agent": "graspa",
}

WORKFLOW_OPTIONS: list[str] = [
    "single_agent",
    "multi_agent",
    "python_relp",
    "graspa",
    "mock_agent",
]


def normalize_workflow_name(value: str) -> str:
    if not value:
        return value
    return WORKFLOW_ALIASES.get(value, value)


def get_model_options(config: Dict[str, Any]) -> list:
    from chemgraph.utils.config_utils import get_model_options_for_nested_config

    return get_model_options_for_nested_config(config)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Configuration page."""
    st.title("\u2699\ufe0f Configuration")
    st.markdown(
        """
    Edit and manage your ChemGraph configuration settings.
    Changes only take effect when you click **Save Configuration**.
    """
    )

    # Ensure config exists in session state
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    # Work on a draft copy so widgets never mutate the live config.
    # The draft is written back to st.session_state.config only on Save.
    if "_config_draft" not in st.session_state:
        st.session_state._config_draft = copy.deepcopy(st.session_state.config)
    draft = st.session_state._config_draft

    # ----- Tabs -----
    tab1, tab2, tab3 = st.tabs(
        ["\U0001f527 General Settings", "\U0001f517 API Settings", "\U0001f4dd Raw TOML"]
    )

    with tab1:
        _render_general_settings(draft)

    with tab2:
        _render_api_settings(draft)

    with tab3:
        _render_raw_toml(draft)

    # ----- Action buttons -----
    _render_action_buttons(draft)

    # ----- Summary -----
    _render_config_summary(draft)


# ---------------------------------------------------------------------------
# Internal renderers
# ---------------------------------------------------------------------------


def _render_general_settings(config: dict) -> None:
    st.subheader("General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model & Workflow**")
        model_options = get_model_options(config)
        config["general"]["model"] = st.selectbox(
            "Model",
            model_options,
            index=(
                model_options.index(config["general"]["model"])
                if config["general"]["model"] in model_options
                else 0
            ),
            key="config_model",
        )
        config_custom_model = st.text_input(
            "Custom model ID (optional)",
            value="",
            key="config_custom_model",
            help="Enter any provider/model identifier not listed above.",
        ).strip()
        if config_custom_model:
            config["general"]["model"] = config_custom_model

        config["general"]["workflow"] = normalize_workflow_name(
            config["general"]["workflow"]
        )
        config["general"]["workflow"] = st.selectbox(
            "Workflow",
            WORKFLOW_OPTIONS,
            index=(
                WORKFLOW_OPTIONS.index(config["general"]["workflow"])
                if config["general"]["workflow"] in WORKFLOW_OPTIONS
                else 0
            ),
            key="config_workflow",
        )

        config["general"]["output"] = st.selectbox(
            "Output Format",
            ["state", "last_message"],
            index=(
                ["state", "last_message"].index(config["general"]["output"])
                if config["general"]["output"] in ["state", "last_message"]
                else 0
            ),
            key="config_output",
        )

        config["general"]["structured"] = st.checkbox(
            "Structured Output",
            value=config["general"]["structured"],
            key="config_structured",
        )
        config["general"]["report"] = st.checkbox(
            "Generate Report",
            value=config["general"]["report"],
            key="config_report",
        )
        config["general"]["verbose"] = st.checkbox(
            "Verbose Output",
            value=config["general"]["verbose"],
            key="config_verbose",
        )

    with col2:
        st.write("**Execution Settings**")
        config["general"]["thread"] = st.number_input(
            "Thread ID",
            min_value=1,
            max_value=1000,
            value=config["general"]["thread"],
            key="config_thread",
        )
        config["general"]["recursion_limit"] = st.number_input(
            "Recursion Limit",
            min_value=1,
            max_value=100,
            value=config["general"]["recursion_limit"],
            key="config_recursion",
        )

    st.subheader("Chemistry Settings")

    col3, col4 = st.columns(2)

    with col3:
        st.write("**Optimization**")
        config["chemistry"]["optimization"]["method"] = st.selectbox(
            "Method",
            ["BFGS", "L-BFGS-B", "CG", "Newton-CG"],
            index=(
                ["BFGS", "L-BFGS-B", "CG", "Newton-CG"].index(
                    config["chemistry"]["optimization"]["method"]
                )
                if config["chemistry"]["optimization"]["method"]
                in ["BFGS", "L-BFGS-B", "CG", "Newton-CG"]
                else 0
            ),
            key="config_opt_method",
        )
        config["chemistry"]["optimization"]["fmax"] = st.number_input(
            "Force Max (eV/\u00c5)",
            min_value=0.001,
            max_value=1.0,
            value=config["chemistry"]["optimization"]["fmax"],
            format="%.3f",
            key="config_fmax",
        )
        config["chemistry"]["optimization"]["steps"] = st.number_input(
            "Max Steps",
            min_value=1,
            max_value=1000,
            value=config["chemistry"]["optimization"]["steps"],
            key="config_steps",
        )

    with col4:
        st.write("**Calculators**")
        calc_options = [
            "mace_mp",
            "mace_off",
            "mace_anicc",
            "fairchem",
            "aimnet2",
            "emt",
            "tblite",
            "orca",
            "nwchem",
        ]
        config["chemistry"]["calculators"]["default"] = st.selectbox(
            "Default Calculator",
            calc_options,
            index=(
                calc_options.index(config["chemistry"]["calculators"]["default"])
                if config["chemistry"]["calculators"]["default"] in calc_options
                else 0
            ),
            key="config_calc_default",
        )
        config["chemistry"]["calculators"]["fallback"] = st.selectbox(
            "Fallback Calculator",
            calc_options,
            index=(
                calc_options.index(config["chemistry"]["calculators"]["fallback"])
                if config["chemistry"]["calculators"]["fallback"] in calc_options
                else 1
            ),
            key="config_calc_fallback",
        )


def _render_api_settings(config: dict) -> None:
    st.subheader("API Settings")

    st.markdown("**API Keys (Session Only)**")
    st.caption(
        "Keys entered here are applied to this Streamlit session via environment "
        "variables and are not saved to config.toml."
    )
    st.warning(
        "**Shared deployments:** API keys are set as process-wide environment "
        "variables. On multi-user Streamlit servers, keys set here may be "
        "visible to other sessions in the same process. For shared "
        "deployments, configure keys via server-side environment variables "
        "instead.",
        icon="\u26a0\ufe0f",
    )

    key_col1, key_col2 = st.columns(2)
    with key_col1:
        openai_api_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.get("ui_openai_api_key", ""),
            type="password",
            key="ui_openai_api_key_input",
        )
        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            value=st.session_state.get("ui_anthropic_api_key", ""),
            type="password",
            key="ui_anthropic_api_key_input",
        )
    with key_col2:
        gemini_api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.get("ui_gemini_api_key", ""),
            type="password",
            key="ui_gemini_api_key_input",
        )
        groq_api_key = st.text_input(
            "Groq API Key",
            value=st.session_state.get("ui_groq_api_key", ""),
            type="password",
            key="ui_groq_api_key_input",
        )

    key_env_map = {
        "OPENAI_API_KEY": openai_api_key,
        "ANTHROPIC_API_KEY": anthropic_api_key,
        "GEMINI_API_KEY": gemini_api_key,
        "GROQ_API_KEY": groq_api_key,
    }

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        if st.button("Apply API Keys", key="apply_api_keys"):
            applied = []
            for env_name, key_value in key_env_map.items():
                clean_value = key_value.strip()
                if clean_value:
                    os.environ[env_name] = clean_value
                    st.session_state[f"ui_{env_name.lower()}"] = clean_value
                    applied.append(env_name)
            if applied:
                st.success(f"\u2705 Applied keys for: {', '.join(applied)}")
            else:
                st.info("No API keys entered.")
    with action_col2:
        if st.button("Clear Session API Keys", key="clear_api_keys"):
            for env_name in key_env_map:
                os.environ.pop(env_name, None)
                st.session_state.pop(f"ui_{env_name.lower()}", None)
                st.session_state.pop(f"ui_{env_name.lower()}_input", None)
            st.success("\u2705 Cleared session API keys.")
            st.rerun()

    st.markdown("---")
    api_tabs = st.tabs(["OpenAI", "Anthropic", "Google", "Local"])

    with api_tabs[0]:
        config["api"]["openai"]["base_url"] = st.text_input(
            "Base URL",
            value=config["api"]["openai"]["base_url"],
            key="config_openai_url",
        )
        config["api"]["openai"]["argo_user"] = st.text_input(
            "Argo User (optional)",
            value=config["api"]["openai"].get("argo_user", ""),
            key="config_openai_argo_user",
            help="ANL domain username for Argo requests. If blank, ChemGraph falls back to ARGO_USER env var.",
        )
        config["api"]["openai"]["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=config["api"]["openai"]["timeout"],
            key="config_openai_timeout",
        )

    with api_tabs[1]:
        config["api"]["anthropic"]["base_url"] = st.text_input(
            "Base URL",
            value=config["api"]["anthropic"]["base_url"],
            key="config_anthropic_url",
        )
        config["api"]["anthropic"]["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=config["api"]["anthropic"]["timeout"],
            key="config_anthropic_timeout",
        )

    with api_tabs[2]:
        config["api"]["google"]["base_url"] = st.text_input(
            "Base URL",
            value=config["api"]["google"]["base_url"],
            key="config_google_url",
        )
        config["api"]["google"]["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=config["api"]["google"]["timeout"],
            key="config_google_timeout",
        )

    with api_tabs[3]:
        config["api"]["local"]["base_url"] = st.text_input(
            "Base URL",
            value=config["api"]["local"]["base_url"],
            key="config_local_url",
        )
        config["api"]["local"]["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=config["api"]["local"]["timeout"],
            key="config_local_timeout",
        )


def _render_raw_toml(config: dict) -> None:
    st.subheader("Raw TOML Configuration")
    st.markdown(
        """
    Edit the raw TOML configuration directly. Be careful with syntax!
    """
    )

    try:
        config_text = toml.dumps(config)
    except Exception as e:
        st.error(f"Error serializing config: {e}")
        config_text = ""

    edited_config = st.text_area(
        "TOML Content", value=config_text, height=400, key="config_raw_toml"
    )

    if st.button("\U0001f4dd Update from TOML", key="update_from_toml"):
        try:
            new_config = toml.loads(edited_config)
            # Update the draft, not the live config.  The user must still
            # click "Save Configuration" to persist and apply the changes.
            st.session_state._config_draft = new_config
            st.success(
                "\u2705 Draft updated from TOML.  "
                "Click **Save Configuration** to apply."
            )
            st.rerun()
        except Exception as e:
            st.error(f"\u274c Invalid TOML syntax: {e}")


def _render_action_buttons(config: dict) -> None:
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("\U0001f4be Save Configuration", type="primary"):
            # Apply the draft to the live session config, then persist to disk.
            st.session_state.config = copy.deepcopy(config)
            if save_config(st.session_state.config):
                st.success("\u2705 Configuration saved to config.toml!")
            else:
                st.error("\u274c Failed to save configuration")

    with col2:
        if st.button("\U0001f504 Reload Configuration"):
            st.session_state.config = load_config()
            st.session_state._config_draft = copy.deepcopy(st.session_state.config)
            st.success("\u2705 Configuration reloaded!")
            st.rerun()

    with col3:
        if st.button("\U0001f5d1\ufe0f Reset to Defaults"):
            st.session_state.config = get_default_config()
            st.session_state._config_draft = copy.deepcopy(st.session_state.config)
            st.success("\u2705 Configuration reset to defaults!")
            st.rerun()

    with col4:
        try:
            config_download = toml.dumps(config)
            st.download_button(
                "\U0001f4e5 Download TOML",
                config_download,
                "config.toml",
                mime="application/toml",
            )
        except Exception as e:
            st.error(f"Error preparing download: {e}")


def _render_config_summary(config: dict) -> None:
    with st.expander("\U0001f4ca Configuration Summary", expanded=False):
        st.write("**Current Configuration:**")
        st.write(f"- Model: {config['general']['model']}")
        st.write(f"- Workflow: {config['general']['workflow']}")
        st.write("- Temperature: 0.0 (optimized for tool calling)")
        st.write("- Max Tokens: 4000")
        st.write(
            f"- Default Calculator: {config['chemistry']['calculators']['default']}"
        )

        st.write("**Environment Variables:**")
        api_keys = {
            "OPENAI_API_KEY": "OpenAI",
            "ANTHROPIC_API_KEY": "Anthropic",
            "GEMINI_API_KEY": "Google",
            "GROQ_API_KEY": "Groq",
        }
        for env_var, provider in api_keys.items():
            if os.getenv(env_var):
                st.write(f"- {provider}: \u2705 Set")
            else:
                st.write(f"- {provider}: \u274c Not set")
