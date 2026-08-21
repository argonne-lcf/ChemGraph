"""Configuration editor page."""

import copy
import os
from typing import Any, Dict

import streamlit as st
import toml

from ui import providers
from ui.config import get_default_config, load_config, save_config
from ui.endpoint import check_local_model_endpoint
from ui.provider_widgets import apply_api_key, clear_api_key, render_alcf_login

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
    "molecular_docking",
    "single_agent_iri",
    "mock_agent",
]


def normalize_workflow_name(value: str) -> str:
    """Normalize workflow aliases to internal workflow names.

    Parameters
    ----------
    value : str
        Workflow name or alias from configuration/UI state.

    Returns
    -------
    str
        Canonical workflow name.
    """
    if not value:
        return value
    return WORKFLOW_ALIASES.get(value, value)


def get_model_options(config: Dict[str, Any]) -> list:
    """Return model options for the configuration UI.

    Parameters
    ----------
    config : dict[str, Any]
        Nested UI configuration dictionary.

    Returns
    -------
    list
        Model names shown in the model selector.
    """
    from chemgraph.utils.config_utils import get_model_options_for_nested_config

    return get_model_options_for_nested_config(config)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Configuration page."""
    st.title("⚙️ Configuration")
    st.markdown(
        """
    Connect a model provider and manage ChemGraph settings.
    Provider actions apply immediately; other changes take effect when you
    click **Save Configuration**.
    """
    )

    # Ensure config exists in session state
    if "config" not in st.session_state or st.session_state.config is None:
        st.session_state.config = load_config()

    # Work on a draft copy so widgets never mutate the live config.
    # The draft is written back to st.session_state.config only on Save.
    # Whenever the live config changed elsewhere (first-run setup, provider
    # activation, reload), the draft is rebased on it and the widget nonce
    # bumps so stale widget state cannot write old values back.
    live_config = st.session_state.config
    if (
        "_config_draft" not in st.session_state
        or st.session_state.get("_config_draft_base") != live_config
    ):
        st.session_state._config_draft = copy.deepcopy(live_config)
        st.session_state._config_draft_base = copy.deepcopy(live_config)
        st.session_state._config_widget_nonce = (
            st.session_state.get("_config_widget_nonce", 0) + 1
        )
    draft = st.session_state._config_draft

    # ----- Tabs -----
    tab_providers, tab_general, tab_chem, tab_toml = st.tabs(
        [
            "\U0001f50c Providers",
            "\U0001f527 General",
            "\U0001f9ea Chemistry",
            "\U0001f4dd Raw TOML",
        ]
    )

    with tab_providers:
        _render_providers(draft)

    with tab_general:
        _render_general_settings(draft)

    with tab_chem:
        _render_chemistry_settings(draft)

    with tab_toml:
        _render_raw_toml(draft)

    # ----- Action buttons -----
    _render_action_buttons(draft)

    # ----- Summary -----
    _render_config_summary(draft)


# ---------------------------------------------------------------------------
# Provider cards
# ---------------------------------------------------------------------------


def _wkey(base: str) -> str:
    """Return a draft-generation-scoped widget key.

    Parameters
    ----------
    base : str
        Stable widget key base.

    Returns
    -------
    str
        Key suffixed with the draft nonce, so widgets are recreated from
        the draft whenever it is rebased on an externally changed config.
    """
    return f"{base}_{st.session_state.get('_config_widget_nonce', 0)}"


def _activate_provider_model(draft: dict, info, model_name: str) -> None:
    """Set *model_name* as the active model and persist immediately.

    Provider activation is a deliberate action, so unlike other draft
    edits it saves right away.

    Parameters
    ----------
    draft : dict
        Mutable draft configuration dictionary.
    info : providers.ProviderInfo
        Provider being activated.
    model_name : str
        Model to activate.
    """
    draft["general"]["model"] = model_name
    providers.align_base_url_for_provider(draft, info.id)
    st.session_state.config = copy.deepcopy(draft)
    if save_config(st.session_state.config):
        st.toast(f"Now using {model_name}", icon="✅")
    st.rerun()


def _render_providers(draft: dict) -> None:
    """Render one status card per provider.

    Parameters
    ----------
    draft : dict
        Mutable draft configuration dictionary.
    """
    st.caption(
        "Configure at least one way to reach an LLM. "
        "✅ = ready to use, ○ = needs setup."
    )
    st.warning(
        "**Shared deployments:** API keys are set as process-wide "
        "environment variables. On multi-user Streamlit servers they may "
        "be visible to other sessions; configure keys via server-side "
        "environment variables instead.",
        icon="⚠️",
    )

    active_model = draft["general"].get("model", "")
    active_info = providers.provider_for_model(active_model)

    for status in providers.all_provider_statuses(draft):
        info = status.info
        badge = "✅" if status.ready else "○"
        is_active = active_info is not None and active_info.id == info.id
        title = f"{badge} {info.icon} {info.label}"
        if is_active:
            title += "  • active"
        with st.expander(title, expanded=is_active and not status.ready):
            st.caption(info.help_text)
            if info.auth_kind == "argo":
                _render_argo_card(draft, info, status)
            elif info.auth_kind == "api_key":
                _render_api_key_card(draft, info, status)
            elif info.auth_kind == "globus":
                _render_alcf_card(draft, info, status)
            elif info.auth_kind == "endpoint":
                _render_vllm_card(draft, info, status)
            else:
                _render_local_card(draft, info, status)
            _render_model_picker(draft, info, status, active_model)


def _render_argo_card(draft: dict, info, status) -> None:
    """Render the Argo gateway card (username, no API key)."""
    argo_section = draft["api"].setdefault("argo", {})
    current = (
        argo_section.get("argo_user")
        or argo_section.get("user")
        or draft["api"].get("openai", {}).get("argo_user", "")
    )
    user = st.text_input(
        "ANL username",
        value=current,
        key=_wkey("provider_argo_user"),
        help=(
            "Your Argonne domain username. Requests to the Argo gateway "
            "are attributed to it; no API key is needed."
        ),
    ).strip()
    if user != current:
        argo_section["argo_user"] = user
        st.session_state.config["api"].setdefault("argo", {})[
            "argo_user"
        ] = user
        save_config(st.session_state.config)
        st.rerun()
    st.caption(status.detail)
    _render_endpoint_settings(draft, "argo", key_prefix="argo")


def _render_api_key_card(draft: dict, info, status) -> None:
    """Render a card for an API-key provider."""
    env_var = info.env_var or ""
    key_set = bool(os.environ.get(env_var))
    if key_set:
        st.success(f"${env_var} is set for this session.")
    key_value = st.text_input(
        f"{info.label} API key",
        value="",
        type="password",
        key=f"provider_key_{info.id}",
        help=(
            f"Applied to this Streamlit process as ${env_var}; "
            "not written to config.toml."
        ),
    )
    col_apply, col_clear = st.columns(2)
    with col_apply:
        if st.button("Apply key", key=f"provider_apply_{info.id}"):
            if apply_api_key(env_var, key_value):
                st.rerun()
            else:
                st.info("Enter a key first.")
    with col_clear:
        if key_set and st.button("Clear key", key=f"provider_clear_{info.id}"):
            clear_api_key(env_var)
            st.rerun()
    if info.config_section:
        _render_endpoint_settings(draft, info.config_section, key_prefix=info.id)


def _render_alcf_card(draft: dict, info, status) -> None:
    """Render the ALCF inference card with the in-UI Globus login."""
    render_alcf_login(key_prefix="config")
    st.caption(
        "Models on the Minerva and Metis clusters are routed to their own "
        "endpoints automatically; the URL below is the Sophia default."
    )
    _render_endpoint_settings(draft, "alcf", key_prefix="alcf")


def _render_local_card(draft: dict, info, status) -> None:
    """Render the local/Ollama card with a live reachability probe."""
    _render_endpoint_settings(draft, "local", key_prefix="local")
    base_url = draft["api"].get("local", {}).get("base_url")
    probe = check_local_model_endpoint(base_url)
    if probe["ok"]:
        st.success(f"Endpoint: {probe['message']}")
    else:
        st.error(f"Endpoint: {probe['message']}")


def _render_vllm_card(draft: dict, info, status) -> None:
    """Render custom OpenAI-compatible endpoint settings."""
    api = draft.setdefault("api", {})
    section = api.get("vllm", {})
    if not isinstance(section, dict):
        section = {}
    base_url = st.text_input(
        "Base URL",
        value=section.get("base_url", ""),
        key=_wkey("endpoint_url_vllm"),
        help="OpenAI-compatible API root, for example http://localhost:8000/v1.",
    )
    _update_vllm_config(api, base_url)
    st.caption(status.detail)


def _update_vllm_config(api: dict, base_url: str) -> None:
    """Preserve an absent legacy vLLM section until a URL is supplied."""
    if "vllm" in api or base_url.strip():
        api.setdefault("vllm", {})["base_url"] = base_url


def _render_endpoint_settings(draft: dict, section: str, key_prefix: str) -> None:
    """Render base-URL/timeout inputs for one ``[api.*]`` section.

    Parameters
    ----------
    draft : dict
        Mutable draft configuration dictionary.
    section : str
        Key under ``config["api"]``.
    key_prefix : str
        Unique widget-key prefix.
    """
    api_section = draft["api"].setdefault(section, {})
    with st.popover("Endpoint settings"):
        api_section["base_url"] = st.text_input(
            "Base URL",
            value=api_section.get("base_url", ""),
            key=_wkey(f"endpoint_url_{key_prefix}"),
        )
        api_section["timeout"] = st.number_input(
            "Timeout (seconds)",
            min_value=1,
            max_value=300,
            value=int(api_section.get("timeout", 30)),
            key=_wkey(f"endpoint_timeout_{key_prefix}"),
        )


def _render_model_picker(draft: dict, info, status, active_model: str) -> None:
    """Render the per-provider model selector and activation button."""
    st.markdown("---")
    col_model, col_use = st.columns([3, 1], vertical_alignment="bottom")
    with col_model:
        if info.models:
            options = list(info.models)
            index = (
                options.index(active_model) if active_model in options else 0
            )
            selected = st.selectbox(
                "Model",
                options,
                index=index,
                key=_wkey(f"provider_model_{info.id}"),
            )
        else:
            selected = st.text_input(
                "Model",
                value=(
                    active_model
                    if providers.provider_for_model(active_model) is not None
                    and providers.provider_for_model(active_model).id == info.id
                    else info.default_model
                ),
                key=_wkey(f"provider_model_{info.id}"),
            ).strip()
    with col_use:
        if st.button(
            "Use",
            key=f"provider_use_{info.id}",
            disabled=not status.ready or not selected,
            help=None if status.ready else status.detail,
            use_container_width=True,
        ):
            _activate_provider_model(draft, info, selected)


# ---------------------------------------------------------------------------
# General / chemistry / raw TOML tabs
# ---------------------------------------------------------------------------


def _render_general_settings(config: dict) -> None:
    """Render and update general configuration widgets.

    Parameters
    ----------
    config : dict
        Mutable draft configuration dictionary.
    """
    st.subheader("General Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Model & Workflow**")
        model_options = get_model_options(config)
        current_model = config["general"]["model"]
        # Keep whatever is active selectable, even when it is not in the
        # curated list (custom IDs, prefix-routed Groq models, ...).
        if current_model and current_model not in model_options:
            model_options = [current_model] + model_options
        config["general"]["model"] = st.selectbox(
            "Model",
            model_options,
            index=(
                model_options.index(current_model)
                if current_model in model_options
                else 0
            ),
            key=_wkey("config_model"),
        )
        custom_model = st.text_input(
            "Custom model ID (optional)",
            value="",
            key=_wkey("config_custom_model"),
            help="Enter any provider/model identifier not listed above.",
        ).strip()
        if st.button(
            "Apply custom model",
            key="config_custom_model_apply",
            disabled=not custom_model,
        ):
            config["general"]["model"] = custom_model
            # Recreate the widgets from the draft so the sticky selectbox
            # state cannot overwrite the custom model on the next render.
            st.session_state._config_widget_nonce = (
                st.session_state.get("_config_widget_nonce", 0) + 1
            )
            st.rerun()

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
            key=_wkey("config_workflow"),
        )

        config["general"]["output"] = st.selectbox(
            "Output Format",
            ["state", "last_message"],
            index=(
                ["state", "last_message"].index(config["general"]["output"])
                if config["general"]["output"] in ["state", "last_message"]
                else 0
            ),
            key=_wkey("config_output"),
        )

        config["general"]["structured"] = st.checkbox(
            "Structured Output",
            value=config["general"]["structured"],
            key=_wkey("config_structured"),
        )
        config["general"]["report"] = st.checkbox(
            "Generate Report",
            value=config["general"]["report"],
            key=_wkey("config_report"),
        )
        config["general"]["human_supervised"] = st.checkbox(
            "Human Supervised",
            value=config["general"].get("human_supervised", False),
            key=_wkey("config_human_supervised"),
            help="Enable the ask_human tool so the agent can pause and request human input.",
        )
        config["general"]["verbose"] = st.checkbox(
            "Verbose Output",
            value=config["general"]["verbose"],
            key=_wkey("config_verbose"),
        )

    with col2:
        st.write("**Execution Settings**")
        config["general"]["thread"] = st.number_input(
            "Thread ID",
            min_value=1,
            max_value=1000,
            value=config["general"]["thread"],
            key=_wkey("config_thread"),
        )
        config["general"]["recursion_limit"] = st.number_input(
            "Recursion Limit",
            min_value=1,
            max_value=100,
            value=config["general"]["recursion_limit"],
            key=_wkey("config_recursion"),
        )


def _render_chemistry_settings(config: dict) -> None:
    """Render and update chemistry configuration widgets.

    Parameters
    ----------
    config : dict
        Mutable draft configuration dictionary.
    """
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
            key=_wkey("config_opt_method"),
        )
        config["chemistry"]["optimization"]["fmax"] = st.number_input(
            "Force Max (eV/Å)",
            min_value=0.001,
            max_value=1.0,
            value=config["chemistry"]["optimization"]["fmax"],
            format="%.3f",
            key=_wkey("config_fmax"),
        )
        config["chemistry"]["optimization"]["steps"] = st.number_input(
            "Max Steps",
            min_value=1,
            max_value=1000,
            value=config["chemistry"]["optimization"]["steps"],
            key=_wkey("config_steps"),
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
            key=_wkey("config_calc_default"),
        )
        config["chemistry"]["calculators"]["fallback"] = st.selectbox(
            "Fallback Calculator",
            calc_options,
            index=(
                calc_options.index(config["chemistry"]["calculators"]["fallback"])
                if config["chemistry"]["calculators"]["fallback"] in calc_options
                else 1
            ),
            key=_wkey("config_calc_fallback"),
        )


def _render_raw_toml(config: dict) -> None:
    """Render raw TOML editor for the draft configuration.

    Parameters
    ----------
    config : dict
        Mutable draft configuration dictionary.
    """
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
        "TOML Content", value=config_text, height=400, key=_wkey("config_raw_toml")
    )

    if st.button("\U0001f4dd Update from TOML", key="update_from_toml"):
        try:
            new_config = toml.loads(edited_config)
            # Update the draft, not the live config.  The user must still
            # click "Save Configuration" to persist and apply the changes.
            st.session_state._config_draft = new_config
            st.success(
                "✅ Draft updated from TOML.  "
                "Click **Save Configuration** to apply."
            )
            st.rerun()
        except Exception as e:
            st.error(f"❌ Invalid TOML syntax: {e}")


def _render_action_buttons(config: dict) -> None:
    """Render save/reload/reset/download configuration actions.

    Parameters
    ----------
    config : dict
        Mutable draft configuration dictionary.
    """
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("\U0001f4be Save Configuration", type="primary"):
            # Apply the draft to the live session config, then persist to disk.
            st.session_state.config = copy.deepcopy(config)
            if save_config(st.session_state.config):
                st.success("✅ Configuration saved to config.toml!")
            else:
                st.error("❌ Failed to save configuration")

    with col2:
        if st.button("\U0001f504 Reload Configuration"):
            st.session_state.config = load_config()
            st.session_state._config_draft = copy.deepcopy(st.session_state.config)
            st.success("✅ Configuration reloaded!")
            st.rerun()

    with col3:
        if st.button("\U0001f5d1️ Reset to Defaults"):
            st.session_state.config = get_default_config()
            st.session_state._config_draft = copy.deepcopy(st.session_state.config)
            st.success("✅ Configuration reset to defaults!")
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
    """Render a compact summary of the draft configuration.

    Parameters
    ----------
    config : dict
        Draft configuration dictionary.
    """
    with st.expander("\U0001f4ca Configuration Summary", expanded=False):
        st.write("**Current Configuration:**")
        st.write(f"- Model: {config['general']['model']}")
        st.write(f"- Workflow: {config['general']['workflow']}")
        st.write(
            f"- Default Calculator: {config['chemistry']['calculators']['default']}"
        )

        st.write("**Providers:**")
        for status in providers.all_provider_statuses(config):
            mark = "✅" if status.ready else "❌"
            st.write(f"- {status.info.label}: {mark} {status.detail}")
