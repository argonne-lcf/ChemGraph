from __future__ import annotations

from chemgraph.models.aurora_endpoints import (
    _normalize_aurora_model,
    load_aurora_model,
)
from chemgraph.models.loader import load_chat_model
from chemgraph.models.supported_models import (
    AURORA_DEFAULT_BASE_URL,
    all_supported_models,
    supported_aurora_models,
)
from chemgraph.utils.config_utils import (
    get_base_url_for_model_from_flat_config,
    get_base_url_for_model_from_nested_config,
)

AURORA_MODEL = "aurora:gpt-oss-120b"
CUSTOM_URL = "http://x4000c0s0b0n0:8000/v1"


def test_every_aurora_model_carries_the_prefix():
    assert all(m.startswith("aurora:") for m in supported_aurora_models)


def test_supported_aurora_models_has_no_duplicates():
    assert len(supported_aurora_models) == len(set(supported_aurora_models))


def test_aurora_models_are_in_all_supported_models():
    assert set(supported_aurora_models) <= set(all_supported_models)


def test_normalize_strips_the_prefix():
    assert _normalize_aurora_model(AURORA_MODEL) == "gpt-oss-120b"
    assert _normalize_aurora_model("aurora:nemotron-4-340b") == "nemotron-4-340b"


def test_normalize_leaves_unprefixed_names_alone():
    assert _normalize_aurora_model("gpt-oss-120b") == "gpt-oss-120b"


def test_base_url_falls_back_to_the_aurora_default():
    assert (
        get_base_url_for_model_from_nested_config(AURORA_MODEL, {})
        == AURORA_DEFAULT_BASE_URL
    )
    assert (
        get_base_url_for_model_from_flat_config(AURORA_MODEL, {})
        == AURORA_DEFAULT_BASE_URL
    )


def test_base_url_honours_configured_aurora_url():
    assert (
        get_base_url_for_model_from_nested_config(
            AURORA_MODEL, {"api": {"aurora": {"base_url": CUSTOM_URL}}}
        )
        == CUSTOM_URL
    )
    assert (
        get_base_url_for_model_from_flat_config(
            AURORA_MODEL, {"api_aurora_base_url": CUSTOM_URL}
        )
        == CUSTOM_URL
    )


def test_uncurated_served_id_still_routes_by_prefix():
    # Dispatch is by prefix, not list membership: an id not in the discovery
    # list still resolves to the aurora endpoint.
    assert (
        get_base_url_for_model_from_flat_config("aurora:some-served-id", {})
        == AURORA_DEFAULT_BASE_URL
    )


def test_load_aurora_model_strips_prefix_and_uses_base_url():
    llm = load_aurora_model(AURORA_MODEL, base_url=CUSTOM_URL, api_key="dummy")
    assert llm.openai_api_base == CUSTOM_URL
    assert llm.model_name == "gpt-oss-120b"


def test_load_aurora_model_defaults_base_url_and_dummy_key(monkeypatch):
    monkeypatch.delenv("AURORA_BASE_URL", raising=False)
    monkeypatch.delenv("AURORA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = load_aurora_model(AURORA_MODEL)
    assert llm.openai_api_base == AURORA_DEFAULT_BASE_URL
    assert llm.model_name == "gpt-oss-120b"


def test_env_base_url_is_used_when_no_explicit_url(monkeypatch):
    monkeypatch.setenv("AURORA_BASE_URL", CUSTOM_URL)
    llm = load_aurora_model("aurora:nemotron-3-ultra", api_key="dummy")
    assert llm.openai_api_base == CUSTOM_URL
    assert llm.model_name == "nemotron-3-ultra"


def test_loader_dispatches_aurora_prefix_to_chatopenai():
    llm = load_chat_model(AURORA_MODEL, base_url=CUSTOM_URL, api_key="dummy")
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.openai_api_base == CUSTOM_URL
    assert llm.model_name == "gpt-oss-120b"


def test_lm_settings_extract_aurora_base_url_from_cli_toml(tmp_path):
    from chemgraph.models.settings import load_lm_settings

    cfg = tmp_path / "aurora.toml"
    cfg.write_text(
        "\n".join(
            [
                "[general]",
                f'model = "{AURORA_MODEL}"',
                "[api.aurora]",
                f'base_url = "{CUSTOM_URL}"',
            ]
        ),
        encoding="utf-8",
    )
    settings = load_lm_settings(cfg)
    assert settings.model == AURORA_MODEL
    assert settings.base_url == CUSTOM_URL
