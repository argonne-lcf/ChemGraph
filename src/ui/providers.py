"""Provider registry and readiness checks for the ChemGraph UI.

One place that knows, for every way ChemGraph can reach an LLM, how to
tell whether it is usable right now and which models it serves.  Used by
the Configuration page (provider cards) and the first-run setup on the
main page.  Streamlit-free and network-free so it can be unit-tested;
live endpoint probes stay in the pages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from chemgraph.models.supported_models import (
    ARGO_DEFAULT_BASE_URL,
    supported_alcf_models,
    supported_anthropic_models,
    supported_argo_models,
    supported_gemini_models,
    supported_ollama_models,
    supported_openai_models,
    supported_openrouter_models,
)
from chemgraph.utils.config_utils import get_argo_user_from_nested_config

from ui import alcf_auth

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

ARGO = "argo"
OPENAI = "openai"
ANTHROPIC = "anthropic"
GOOGLE = "google"
GROQ = "groq"
OPENROUTER = "openrouter"
ALCF = "alcf"
OLLAMA = "ollama"


@dataclass(frozen=True)
class ProviderInfo:
    """Static description of one LLM provider."""

    id: str
    label: str
    icon: str
    # How the provider authenticates: "argo" (username), "api_key",
    # "globus" (ALCF), or "none" (local server).
    auth_kind: str
    # Environment variable carrying the credential, when applicable.
    env_var: Optional[str]
    # Section under config["api"] holding base_url/timeout for this provider.
    config_section: Optional[str]
    default_model: str
    models: tuple
    help_text: str


PROVIDERS: tuple[ProviderInfo, ...] = (
    ProviderInfo(
        id=ARGO,
        label="Argo (Argonne)",
        icon="\U0001f3db",
        auth_kind="argo",
        env_var="ARGO_USER",
        config_section="openai",
        default_model="argo:gpt-4o",
        models=tuple(supported_argo_models),
        help_text=(
            "Argonne's internal LLM gateway. Needs only your ANL domain "
            "username -- no API key -- but works only on the lab network "
            "or VPN."
        ),
    ),
    ProviderInfo(
        id=OPENAI,
        label="OpenAI",
        icon="\U0001f511",
        auth_kind="api_key",
        env_var="OPENAI_API_KEY",
        config_section="openai",
        default_model="gpt-4o-mini",
        models=tuple(supported_openai_models),
        help_text="Direct OpenAI API access with your own API key.",
    ),
    ProviderInfo(
        id=ANTHROPIC,
        label="Anthropic",
        icon="\U0001f511",
        auth_kind="api_key",
        env_var="ANTHROPIC_API_KEY",
        config_section="anthropic",
        default_model="claude-sonnet-4-20250514",
        models=tuple(supported_anthropic_models),
        help_text="Claude models with your own Anthropic API key.",
    ),
    ProviderInfo(
        id=GOOGLE,
        label="Google Gemini",
        icon="\U0001f511",
        auth_kind="api_key",
        env_var="GEMINI_API_KEY",
        config_section="google",
        default_model="gemini-2.5-flash",
        models=tuple(supported_gemini_models),
        help_text="Gemini models with your own Google AI Studio key.",
    ),
    ProviderInfo(
        id=GROQ,
        label="Groq",
        icon="\U0001f511",
        auth_kind="api_key",
        env_var="GROQ_API_KEY",
        config_section="groq",
        default_model="groq:llama-3.3-70b-versatile",
        models=(),
        help_text=(
            "Fast open-model inference. Use any model as "
            "'groq:<model-id>' from console.groq.com/docs/models."
        ),
    ),
    ProviderInfo(
        id=OPENROUTER,
        label="OpenRouter",
        icon="\U0001f511",
        auth_kind="api_key",
        env_var="OPENROUTER_API_KEY",
        config_section="openrouter",
        default_model="openrouter:deepseek/deepseek-v4-flash",
        models=tuple(supported_openrouter_models),
        help_text=(
            "One key for many hosted models. Any slug from "
            "openrouter.ai/models works as 'openrouter:<slug>'."
        ),
    ),
    ProviderInfo(
        id=ALCF,
        label="ALCF Inference (Globus)",
        icon="\U0001f310",
        auth_kind="globus",
        env_var=alcf_auth.TOKEN_ENV,
        config_section="alcf",
        default_model="alcf:meta-llama/Llama-3.3-70B-Instruct",
        models=tuple(supported_alcf_models),
        help_text=(
            "ALCF-hosted open models (Sophia/Minerva/Metis clusters). "
            "Log in with your Globus account -- requires an active ALCF "
            "allocation."
        ),
    ),
    ProviderInfo(
        id=OLLAMA,
        label="Local (Ollama)",
        icon="\U0001f4bb",
        auth_kind="none",
        env_var=None,
        config_section="local",
        default_model="llama3.2",
        models=tuple(supported_ollama_models),
        help_text=(
            "Models served by a local Ollama (or other OpenAI-compatible) "
            "server. No credentials needed."
        ),
    ),
)


@dataclass
class ProviderStatus:
    """Readiness of one provider given current config + environment."""

    info: ProviderInfo
    ready: bool
    detail: str


def get_provider(provider_id: str) -> Optional[ProviderInfo]:
    """Return the provider description for *provider_id*, if known."""
    for info in PROVIDERS:
        if info.id == provider_id:
            return info
    return None


def provider_status(
    info: ProviderInfo, config: Dict[str, Any]
) -> ProviderStatus:
    """Evaluate whether one provider is usable right now.

    Parameters
    ----------
    info : ProviderInfo
        Provider description.
    config : dict[str, Any]
        Nested UI configuration.

    Returns
    -------
    ProviderStatus
        Readiness plus a short human-readable reason.
    """
    if info.auth_kind == "argo":
        user = get_argo_user_from_nested_config(config) or os.environ.get(
            "ARGO_USER"
        )
        if user:
            return ProviderStatus(
                info, True, f"Argo user '{user}' (ANL network/VPN required)."
            )
        return ProviderStatus(info, False, "Set your ANL username.")

    if info.auth_kind == "api_key":
        if os.environ.get(info.env_var or ""):
            return ProviderStatus(info, True, f"${info.env_var} is set.")
        return ProviderStatus(info, False, f"Set ${info.env_var}.")

    if info.auth_kind == "globus":
        status = alcf_auth.token_status()
        ready = status["state"] in ("env", "valid", "refreshable")
        return ProviderStatus(info, ready, status["detail"])

    # Local server: configuration alone cannot prove readiness; the page
    # adds a live reachability probe.
    return ProviderStatus(info, True, "Requires a running local server.")


def all_provider_statuses(config: Dict[str, Any]) -> list[ProviderStatus]:
    """Return statuses for every known provider."""
    return [provider_status(info, config) for info in PROVIDERS]


def any_provider_ready(config: Dict[str, Any]) -> bool:
    """Return whether at least one credentialed provider is usable.

    The local server is excluded: it is always nominally "ready" and
    would defeat first-run detection.
    """
    return any(
        status.ready
        for status in all_provider_statuses(config)
        if status.info.auth_kind != "none"
    )


def align_base_url_for_provider(config: Dict[str, Any], provider_id: str) -> None:
    """Repoint the shared ``[api.openai]`` URL when activating Argo or OpenAI.

    Both providers route through ``config["api"]["openai"]["base_url"]``:
    Argo models need the Argonne gateway there, plain OpenAI models need
    the public endpoint. Activating either one must fix the URL, or the
    other provider's previous setting silently misroutes the requests.

    Parameters
    ----------
    config : dict[str, Any]
        Nested UI configuration (mutated in place).
    provider_id : str
        Activated provider ID; other providers are left untouched.
    """
    section = config.setdefault("api", {}).setdefault("openai", {})
    if provider_id == ARGO:
        section["base_url"] = ARGO_DEFAULT_BASE_URL
    elif provider_id == OPENAI:
        section["base_url"] = OPENAI_DEFAULT_BASE_URL


def provider_for_model(model_name: str) -> Optional[ProviderInfo]:
    """Return the provider that serves *model_name*, if identifiable.

    Mirrors the dispatch order of the model loader: prefixes first, then
    curated lists. Unknown names default to OpenAI-compatible routing.

    Parameters
    ----------
    model_name : str
        Model identifier.

    Returns
    -------
    ProviderInfo or None
        Matching provider description.
    """
    if not model_name:
        return None
    prefix_map = {
        "argo:": ARGO,
        "groq:": GROQ,
        "openrouter:": OPENROUTER,
        "alcf:": ALCF,
    }
    for prefix, provider_id in prefix_map.items():
        if model_name.startswith(prefix):
            return get_provider(provider_id)
    if model_name in supported_openai_models:
        return get_provider(OPENAI)
    if model_name in supported_anthropic_models:
        return get_provider(ANTHROPIC)
    if model_name in supported_gemini_models:
        return get_provider(GOOGLE)
    if model_name in supported_ollama_models:
        return get_provider(OLLAMA)
    return get_provider(OPENAI)
