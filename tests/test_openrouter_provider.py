from __future__ import annotations

import io
import sys

import pytest

from chemgraph.cli.commands import check_api_keys
from chemgraph.models.loader import load_chat_model
from chemgraph.models.openrouter import (
    OPENROUTER_DEFAULT_MAX_TOKENS,
    load_openrouter_model,
)
from chemgraph.models.supported_models import (
    OPENROUTER_DEFAULT_BASE_URL,
    all_supported_models,
    supported_openrouter_models,
)
from chemgraph.utils.config_utils import (
    get_base_url_for_model_from_flat_config,
    get_base_url_for_model_from_nested_config,
)

CURATED_MODEL = "openrouter:moonshotai/kimi-k3"
UNCURATED_MODEL = "openrouter:some/unknown-model"
# The value shipped in the repo's config.toml -- an OpenRouter model must never
# resolve to it.
ARGO_BASE_URL = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"


def test_curated_models_carry_the_openrouter_prefix():
    assert supported_openrouter_models
    for model in supported_openrouter_models:
        assert model.startswith("openrouter:")


def test_curated_models_are_in_all_supported_models():
    assert set(supported_openrouter_models) <= set(all_supported_models)


def test_flat_config_does_not_route_openrouter_to_argo():
    config = {"api_openai_base_url": ARGO_BASE_URL}

    assert (
        get_base_url_for_model_from_flat_config(CURATED_MODEL, config)
        == OPENROUTER_DEFAULT_BASE_URL
    )


def test_nested_config_does_not_route_openrouter_to_argo():
    config = {"api": {"openai": {"base_url": ARGO_BASE_URL}}}

    assert (
        get_base_url_for_model_from_nested_config(CURATED_MODEL, config)
        == OPENROUTER_DEFAULT_BASE_URL
    )


def test_configured_openrouter_base_url_is_honoured():
    custom = "https://example.invalid/api/v1"

    assert (
        get_base_url_for_model_from_nested_config(
            CURATED_MODEL,
            {
                "api": {
                    "openai": {"base_url": ARGO_BASE_URL},
                    "openrouter": {"base_url": custom},
                }
            },
        )
        == custom
    )
    assert (
        get_base_url_for_model_from_flat_config(
            CURATED_MODEL,
            {"api_openai_base_url": ARGO_BASE_URL, "api_openrouter_base_url": custom},
        )
        == custom
    )


def test_uncurated_openrouter_slug_still_resolves():
    """Dispatch is by prefix, not list membership.

    If either resolver used ``in supported_openrouter_models`` instead, an
    uncurated slug would fall through to the ``[api.openai]`` catch-all.
    """
    assert (
        get_base_url_for_model_from_nested_config(
            UNCURATED_MODEL, {"api": {"openai": {"base_url": ARGO_BASE_URL}}}
        )
        == OPENROUTER_DEFAULT_BASE_URL
    )
    assert (
        get_base_url_for_model_from_flat_config(
            UNCURATED_MODEL, {"api_openai_base_url": ARGO_BASE_URL}
        )
        == OPENROUTER_DEFAULT_BASE_URL
    )


def test_load_openrouter_model_strips_the_prefix():
    llm = load_openrouter_model(CURATED_MODEL, api_key="dummy")

    assert llm.model_name == "moonshotai/kimi-k3"
    assert llm.openai_api_base == OPENROUTER_DEFAULT_BASE_URL
    assert llm.max_tokens == OPENROUTER_DEFAULT_MAX_TOKENS


def test_explicit_base_url_overrides_the_default():
    custom = "http://127.0.0.1:8000/v1"
    llm = load_openrouter_model(CURATED_MODEL, base_url=custom, api_key="dummy")

    assert llm.openai_api_base == custom


def test_api_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-from-env")

    llm = load_openrouter_model(CURATED_MODEL)

    assert llm.openai_api_key.get_secret_value() == "sk-or-from-env"


def test_openai_api_key_is_never_used_as_a_fallback(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-should-not-leak")
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        load_openrouter_model(CURATED_MODEL)


def test_missing_key_raises_instead_of_prompting_when_not_a_tty(monkeypatch):
    """A getpass() prompt with no terminal blocks forever instead of failing.

    The eval harness runs unattended, where a hang has no traceback and no
    exit code -- strictly worse than a crash.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # StringIO().isatty() is False.
    monkeypatch.setattr(sys, "stdin", io.StringIO())

    with pytest.raises(ValueError, match="openrouter.ai/keys"):
        load_openrouter_model(CURATED_MODEL)


def test_loader_dispatches_openrouter_and_forwards_base_url():
    """Also pins that ``base_url`` is not dropped the way the groq branch drops it."""
    default = load_chat_model(model_name=CURATED_MODEL, api_key="dummy")
    assert default.openai_api_base == OPENROUTER_DEFAULT_BASE_URL

    custom = "https://example.invalid/api/v1"
    forwarded = load_chat_model(
        model_name=CURATED_MODEL, api_key="dummy", base_url=custom
    )
    assert forwarded.openai_api_base == custom


def test_check_api_keys_reports_openrouter(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    ok, message = check_api_keys(CURATED_MODEL)
    assert not ok
    assert "OPENROUTER_API_KEY" in message

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    assert check_api_keys(CURATED_MODEL) == (True, "")


@pytest.mark.parametrize(
    "model",
    ["openrouter:openai/o3", "openrouter:meta-llama/llama-4-maverick"],
)
def test_check_api_keys_is_not_fooled_by_substring_matches(monkeypatch, model):
    """``"o3" in model_lower`` and ``"llama" in model_lower`` are substring tests.

    Without the OpenRouter branch running first, these slugs would be routed to
    the OpenAI key check and the no-key-needed local branch respectively.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-present")

    ok, message = check_api_keys(model)

    assert not ok
    assert "OPENROUTER_API_KEY" in message
