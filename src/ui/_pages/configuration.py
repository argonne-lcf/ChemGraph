"""Configuration editor page."""

import copy
import os
from pathlib import Path
from typing import Any, Dict

import streamlit as st
import toml

from ui import config as ui_config
from ui import providers
from ui.config import (
    get_default_config, load_config, merge_config_defaults,
    resolve_default_calculator, save_config,
)
from ui.endpoint import check_local_model_endpoint
from ui.provider_widgets import (
    apply_api_key, clear_api_key, render_alcf_login, render_codex_login,
)

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
    "deep_agent",
    "python_relp",
    "graspa",
    "molecular_docking",
    "single_agent_iri",
    "mock_agent",
]

#: Session-state flag recording that the user acknowledged the Deep Agent's
#: host-shell access in this browser session (the CLI asks the same
#: question once per process).
DEEPAGENT_ACK_KEY = "deepagent_host_shell_acknowledged"


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


def _save_failure_message() -> str:
    """Describe a failed configuration save with the path and a remedy."""
    reason = ui_config.last_save_error or ui_config.config_path()
    return (
        f"❌ Failed to save configuration ({reason}). Settings apply to this "
        f"session only; set ${ui_config.CONFIG_PATH_ENV} to a writable file to "
        "persist them."
    )


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
    st.caption(f"Configuration file: `{ui_config.config_path()}`")
    save_error = st.session_state.pop("config_save_error", None)
    if save_error:
        st.error(save_error)

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
    tab_providers, tab_general, tab_chem, tab_deep, tab_toml = st.tabs(
        [
            "\U0001f50c Providers",
            "\U0001f527 General",
            "\U0001f9ea Chemistry",
            "\U0001f916 Deep Agent",
            "\U0001f4dd Raw TOML",
        ]
    )

    with tab_providers:
        _render_providers(draft)

    with tab_general:
        _render_general_settings(draft)

    with tab_chem:
        _render_chemistry_settings(draft)

    with tab_deep:
        _render_deepagent_settings(draft)

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
    else:
        st.session_state.config_save_error = _save_failure_message()
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
            elif info.auth_kind == "codex":
                _render_codex_card(draft, info, status)
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
        if not save_config(st.session_state.config):
            st.session_state.config_save_error = _save_failure_message()
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


def _render_codex_card(draft: dict, info, status) -> None:
    """Render the Codex subscription card with the in-UI device-code login."""
    st.caption(
        "The login is stored by Codex on the machine hosting this UI "
        "(the same login `chemgraph run --model codex:<id>` uses). API-key "
        "logins are refused; sign in with ChatGPT."
    )
    if render_codex_login(key_prefix="config"):
        st.rerun()
    if status.ready and not providers.provider_models(info):
        st.caption(
            "Codex did not return a model catalog for this account; type a "
            "model id below (for example `codex:gpt-5.1-codex`)."
        )


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


_CUSTOM_MODEL_OPTION = "__custom__"


def _codex_model_label(name: str) -> str:
    """Selectbox label for a Codex catalog entry: display name, id and default mark."""
    if name == _CUSTOM_MODEL_OPTION:
        return "Other model id…"
    from ui import codex_auth

    for item in codex_auth.available_models():
        if item["name"] == name:
            label = item["display_name"]
            if item["model"] != item["display_name"]:
                label += f" ({item['model']})"
            if item["is_default"]:
                label += " — default"
            return label
    return name


def _render_model_picker(draft: dict, info, status, active_model: str) -> None:
    """Render the per-provider model selector and activation button."""
    st.markdown("---")
    col_model, col_use = st.columns([3, 1], vertical_alignment="bottom")
    with col_model:
        options = list(providers.provider_models(info))
        if options and info.auth_kind == "codex":
            # Account catalog fetched live; keep an escape hatch for ids
            # Codex does not list (hidden/preview models).
            options.append(_CUSTOM_MODEL_OPTION)
            default_name = providers.default_model_for(info)
            index = (
                options.index(active_model)
                if active_model in options
                else options.index(default_name) if default_name in options else 0
            )
            choice = st.selectbox(
                "Model",
                options,
                index=index,
                format_func=lambda name: _codex_model_label(name),
                key=_wkey(f"provider_model_{info.id}"),
            )
            if choice == _CUSTOM_MODEL_OPTION:
                selected = st.text_input(
                    "Model id",
                    value=(
                        active_model
                        if active_model.startswith("codex:")
                        and active_model not in options
                        else "codex:"
                    ),
                    key=_wkey(f"provider_model_custom_{info.id}"),
                    help="Any model available to the signed-in account, as codex:<model-id>.",
                ).strip()
            else:
                selected = choice
        elif options:
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
            "mace_polar",
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
        calculators = config["chemistry"]["calculators"]
        default = calculators.get("default")
        default_options = [None, *calc_options]
        if default not in default_options:
            default_options.append(default)
        selected = st.selectbox(
            "Default Calculator",
            default_options,
            index=default_options.index(default),
            format_func=lambda value: (
                f"Automatic ({resolve_default_calculator({})})"
                if value is None else value
            ),
            key=_wkey("config_calc_default"),
        )
        if selected is None:
            calculators.pop("default", None)
        else:
            calculators["default"] = selected
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


def _registry_tool_names() -> list[str]:
    """Return the names in the built-in Deep Agent tool catalog."""
    try:
        from chemgraph.registry.tools import ToolRegistry

        return list(ToolRegistry().names())
    except Exception:
        return []


def render_deepagent_acknowledgment(key: str) -> bool:
    """Render the host-shell acknowledgment checkbox and return its state.

    Parameters
    ----------
    key : str
        Unique widget key (the checkbox appears on two pages).

    Returns
    -------
    bool
        Whether the user has acknowledged host-shell access this session.
    """
    acknowledged = st.checkbox(
        "I understand the Deep Agent can run shell commands on this host and "
        "modify files under the workspace; every command and file change "
        "will ask for my approval in the chat.",
        value=bool(st.session_state.get(DEEPAGENT_ACK_KEY, False)),
        key=key,
    )
    st.session_state[DEEPAGENT_ACK_KEY] = bool(acknowledged)
    return bool(acknowledged)


def _render_deepagent_settings(config: dict) -> None:
    """Render Deep Agent workspace, skills and tool-catalog settings.

    The keys match what ``chemgraph run --config`` reads, so a config.toml
    saved here also drives the CLI.

    Parameters
    ----------
    config : dict
        Mutable draft configuration dictionary.
    """
    general = config["general"]
    st.subheader("Deep Agent")
    st.markdown(
        "The experimental **deep_agent** workflow gives the model a shell and "
        "file tools rooted at a workspace directory, plus on-demand chemistry "
        "tools and skills. Select `deep_agent` as the workflow on the General "
        "tab to use these settings."
    )
    st.warning(
        "The shell is **not** confined to the workspace directory. Shell "
        "commands and file mutations pause for your approval in the chat "
        "(Approve / Reject buttons); approvals cannot be disabled in the UI.",
        icon="\u26a0\ufe0f",
    )
    render_deepagent_acknowledgment(key="config_deepagent_ack")

    col_ws, col_skills = st.columns(2)
    with col_ws:
        st.write("**Workspace**")
        workspace = st.text_input(
            "Workspace directory",
            value=str(general.get("deepagent_workspace") or ""),
            key=_wkey("config_deepagent_workspace"),
            help=(
                "Directory the agent's file tools are rooted at. Leave empty "
                "to use the directory the UI was launched from."
            ),
        ).strip()
        general["deepagent_workspace"] = workspace
        if workspace:
            path = Path(workspace).expanduser()
            if path.is_dir():
                st.caption(f"Resolved: `{path.resolve()}`")
            else:
                st.error("This directory does not exist.")
        general["deepagent_discover_skills"] = st.checkbox(
            "Discover personal and project skills",
            value=bool(general.get("deepagent_discover_skills", True)),
            key=_wkey("config_deepagent_discover"),
            help=(
                "Also load skills from the personal skills directory and "
                "the workspace's project skills; bundled skills are always "
                "available."
            ),
        )

    with col_skills:
        st.write("**Extra skill directories**")
        current_dirs = general.get("deepagent_skills") or []
        if isinstance(current_dirs, str):
            current_dirs = [current_dirs]
        skills_text = st.text_area(
            "One directory per line",
            value="\n".join(str(item) for item in current_dirs),
            key=_wkey("config_deepagent_skills"),
            height=120,
            help=(
                "Host directories containing skills (each skill is a "
                "subdirectory with a SKILL.md). Mounted in addition to the "
                "bundled skills; equivalent to repeated --deepagent-skill."
            ),
        )
        general["deepagent_skills"] = [
            line.strip() for line in skills_text.splitlines() if line.strip()
        ]
        for item in general["deepagent_skills"]:
            if not Path(item).expanduser().is_dir():
                st.error(f"Skill directory does not exist: `{item}`")

    st.write("**On-demand tool catalog**")
    catalog = _registry_tool_names()
    restrict = st.checkbox(
        "Restrict the catalog to selected tools",
        value="tools" in general,
        key=_wkey("config_deepagent_restrict_tools"),
        help=(
            "By default the agent can discover every built-in registry tool. "
            "Restricting to a subset mirrors --tool; selecting none disables "
            "discovery. This edits the shared `tools` key, which the CLI also "
            "uses for interactive main_agent tool opt-in; unticking removes it."
        ),
    )
    if restrict:
        selected_default = [
            name for name in (general.get("tools") or []) if name in catalog
        ]
        general["tools"] = st.multiselect(
            "Tools",
            catalog,
            default=selected_default,
            key=_wkey("config_deepagent_tools"),
        )
    else:
        general.pop("tools", None)


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
            st.session_state._config_draft = merge_config_defaults(new_config)
            st.session_state._config_widget_nonce = (
                st.session_state.get("_config_widget_nonce", 0) + 1
            )
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
                st.success(f"✅ Configuration saved to {ui_config.config_path()}")
            else:
                st.error(_save_failure_message())

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
            f"- Default Calculator: {resolve_default_calculator(config)}"
        )

        st.write("**Providers:**")
        for status in providers.all_provider_statuses(config):
            mark = "✅" if status.ready else "❌"
            st.write(f"- {status.info.label}: {mark} {status.detail}")
