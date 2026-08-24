"""PR 3 regression tests for configuration + auth cleanup (issue #201).

These lock in the behaviors PR 3 introduced on top of the endpoint registry:

- ``LLMSettings`` carries ``reasoning_effort`` and round-trips it through
  ``load_lm_settings`` / mappings.
- The CLI TOML section resolver reads each endpoint's own ``[api.<section>]``
  for both ``base_url`` and ``api_key`` -- Anthropic/Gemini/ALCF keys are no
  longer sourced from ``[api.openai]`` (the bug this PR fixes).
- The Argo OpenAI/Anthropic split resolves to distinct config sections.
- The legacy ``[api.openai]`` base-URL fallback still works for argo/vllm and
  emits a migration warning.
- ``config_utils`` URL ladders and ``check_api_keys`` resolve through endpoint
  metadata rather than per-provider branches.
"""

from __future__ import annotations

import logging

import pytest

from chemgraph.models.settings import (
    LLMSettings,
    _extract_endpoint_from_cli_toml,
    load_lm_settings,
)
from chemgraph.models.supported_models import (
    ALCF_METIS_BASE_URL,
    ALCF_MINERVA_BASE_URL,
    supported_anthropic_models,
    supported_alcf_metis_models,
    supported_alcf_minerva_models,
    supported_gemini_models,
)

ANTHROPIC_MODEL = supported_anthropic_models[0]
GEMINI_MODEL = supported_gemini_models[0]
MINERVA_MODEL = supported_alcf_minerva_models[0]
METIS_MODEL = supported_alcf_metis_models[0]


# ---------------------------------------------------------------------------
# reasoning_effort field
# ---------------------------------------------------------------------------


def test_reasoning_effort_roundtrips_through_llm_settings():
    settings = LLMSettings(model="gpt-4o", reasoning_effort="high")
    assert settings.reasoning_effort == "high"


def test_reasoning_effort_roundtrips_through_load_lm_settings():
    settings = load_lm_settings({"model": "gpt-4o", "reasoning_effort": "medium"})
    assert settings.reasoning_effort == "medium"


def test_reasoning_effort_defaults_to_none():
    assert load_lm_settings({"model": "gpt-4o"}).reasoning_effort is None


def test_reasoning_effort_addition_preserves_existing_positional_arguments():
    settings = LLMSettings(
        "gpt-4o",
        "https://example/v1",
        "key",
        "argo-user",
        None,
        30,
        0.1,
        1024,
        2,
        0.5,
        "academy-user",
    )
    assert settings.timeout_s == 30
    assert settings.user == "argo-user"
    assert settings.reasoning_effort is None


def test_reasoning_effort_roundtrips_through_toml(tmp_path):
    """A [general].reasoning_effort in a TOML file must survive loading."""
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        '[general]\nmodel = "argo:gpt-4o"\nreasoning_effort = "high"\n'
    )
    assert load_lm_settings(str(cfg)).reasoning_effort == "high"


# ---------------------------------------------------------------------------
# Section-resolution bug fix: keys come from the endpoint's own section
# ---------------------------------------------------------------------------


def _cli_toml(model: str) -> dict:
    return {
        "general": {"model": model},
        "api": {
            "openai": {
                "base_url": "https://argo/openai",
                "api_key": "OPENAI_KEY",
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com",
                "api_key": "ANTHROPIC_KEY",
            },
            "google": {
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "api_key": "GEMINI_KEY",
            },
        },
    }


def test_anthropic_key_not_sourced_from_openai_section():
    """Regression: Anthropic keys used to default to [api.openai]."""
    resolved = _extract_endpoint_from_cli_toml(_cli_toml(ANTHROPIC_MODEL))
    assert resolved["api_key"] == "ANTHROPIC_KEY"
    assert resolved["base_url"] == "https://api.anthropic.com"


def test_gemini_key_not_sourced_from_openai_section():
    resolved = _extract_endpoint_from_cli_toml(_cli_toml(GEMINI_MODEL))
    assert resolved["api_key"] == "GEMINI_KEY"
    assert resolved["base_url"] == "https://generativelanguage.googleapis.com/v1beta"


def test_openai_model_still_reads_openai_section():
    resolved = _extract_endpoint_from_cli_toml(_cli_toml("gpt-4o"))
    assert resolved["api_key"] == "OPENAI_KEY"


@pytest.mark.parametrize(
    ("model", "expected_url"),
    [(MINERVA_MODEL, ALCF_MINERVA_BASE_URL), (METIS_MODEL, ALCF_METIS_BASE_URL)],
)
def test_alcf_config_does_not_override_model_cluster(model, expected_url):
    raw = {
        "general": {"model": model},
        "api": {
            "alcf": {
                "base_url": "https://sophia.example/v1",
                "api_key": "ALCF_KEY",
            }
        },
    }
    resolved = _extract_endpoint_from_cli_toml(raw)
    assert resolved["base_url"] == expected_url
    assert resolved["api_key"] == "ALCF_KEY"


# ---------------------------------------------------------------------------
# Argo OpenAI / Anthropic split resolves to distinct sections
# ---------------------------------------------------------------------------


def _argo_toml(model: str) -> dict:
    return {
        "general": {"model": model},
        "api": {
            "argo": {
                "base_url": "https://argo/v1",
                "api_key": "ARGO_KEY",
                "argo_user": "alice",
            },
        },
    }


def test_argo_openai_model_reads_argo_section():
    resolved = _extract_endpoint_from_cli_toml(_argo_toml("argo:gpt-4o"))
    assert resolved["base_url"] == "https://argo/v1"
    assert resolved["api_key"] == "ARGO_KEY"


def test_argo_claude_model_reads_shared_argo_section():
    resolved = _extract_endpoint_from_cli_toml(_argo_toml("argo:claude-opus-5"))
    assert resolved["base_url"] == "https://argo"
    assert resolved["api_key"] == "ARGO_KEY"


# ---------------------------------------------------------------------------
# Legacy [api.openai] base-URL fallback for argo / vllm (one release, warns)
# ---------------------------------------------------------------------------


@pytest.fixture
def _propagate_logs(monkeypatch):
    """ChemGraph loggers set propagate=False; re-enable so caplog sees them."""
    for name in (
        "chemgraph.models.endpoints.configuration",
        "chemgraph.models.endpoints.vllm",
    ):
        monkeypatch.setattr(logging.getLogger(name), "propagate", True)


def test_argo_falls_back_to_legacy_openai_base_url_with_warning(caplog, _propagate_logs):
    raw = {
        "general": {"model": "argo:gpt-4o"},
        "api": {"openai": {"base_url": "https://legacy/argo"}},
    }
    with caplog.at_level(logging.WARNING):
        resolved = _extract_endpoint_from_cli_toml(raw)
    assert resolved["base_url"] == "https://legacy/argo"
    assert any("legacy [api.openai]" in r.message for r in caplog.records)


def test_argo_user_falls_back_through_sections():
    raw = {
        "general": {"model": "argo:gpt-4o"},
        "api": {"openai": {"argo_user": "legacy_user"}},
    }
    assert _extract_endpoint_from_cli_toml(raw)["argo_user"] == "legacy_user"

    raw["api"]["argo"] = {"argo_user": "canonical_user"}
    assert _extract_endpoint_from_cli_toml(raw)["argo_user"] == "canonical_user"


def test_argo_canonical_section_beats_legacy_section():
    raw = {
        "general": {"model": "argo:gpt-4o"},
        "api": {
            "argo": {"base_url": "https://canonical/argo/v1"},
            "openai": {"base_url": "https://legacy/argo/v1"},
        },
    }
    assert _extract_endpoint_from_cli_toml(raw)["base_url"] == (
        "https://canonical/argo/v1"
    )


# ---------------------------------------------------------------------------
# config_utils URL ladders resolve through endpoint metadata
# ---------------------------------------------------------------------------


def test_nested_config_routes_shared_argo_section():
    from chemgraph.utils.config_utils import (
        get_base_url_for_model_from_nested_config as nested,
    )

    cfg = {"api": {"argo": {"base_url": "https://argo/v1"}}}
    assert nested("argo:gpt-4o", cfg) == "https://argo/v1"
    assert nested("argo:claude-opus-5", cfg) == "https://argo"


def test_nested_config_anthropic_reads_own_section():
    from chemgraph.utils.config_utils import (
        get_base_url_for_model_from_nested_config as nested,
    )

    cfg = {"api": {"anthropic": {"base_url": "https://api.anthropic.com"}}}
    assert nested(ANTHROPIC_MODEL, cfg) == "https://api.anthropic.com"


def test_flat_config_argo_legacy_fallback_warns(caplog, _propagate_logs):
    from chemgraph.utils.config_utils import (
        get_base_url_for_model_from_flat_config as flat,
    )

    with caplog.at_level(logging.WARNING):
        url = flat("argo:gpt-4o", {"api_openai_base_url": "https://legacy/argo/v1"})
    assert url == "https://legacy/argo/v1"
    assert any("legacy [api.openai]" in r.message for r in caplog.records)


def test_empty_vllm_section_suppresses_openai_fallback(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    raw = {
        "general": {"model": "custom-model"},
        "api": {
            "openai": {"base_url": "https://api.openai.com/v1"},
            "vllm": {"base_url": ""},
        },
    }
    assert _extract_endpoint_from_cli_toml(raw)["base_url"] is None


def test_empty_vllm_section_still_allows_environment(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "https://env.example/v1")
    raw = {
        "general": {"model": "custom-model"},
        "api": {
            "openai": {"base_url": "https://api.openai.com/v1"},
            "vllm": {"base_url": ""},
        },
    }
    assert _extract_endpoint_from_cli_toml(raw)["base_url"] == (
        "https://env.example/v1"
    )


def test_vllm_canonical_url_beats_legacy_and_environment(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "https://env.example/v1")
    raw = {
        "general": {"model": "custom-model"},
        "api": {
            "openai": {"base_url": "https://legacy.example/v1"},
            "vllm": {"base_url": "https://canonical.example/v1"},
        },
    }
    assert _extract_endpoint_from_cli_toml(raw)["base_url"] == (
        "https://canonical.example/v1"
    )


def test_vllm_key_never_comes_from_openai_config_section(monkeypatch):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    raw = {
        "general": {"model": "custom-model"},
        "api": {
            "openai": {"api_key": "OPENAI_CONFIG_KEY"},
            "vllm": {
                "base_url": "https://canonical.example/v1",
                "api_key": "VLLM_CONFIG_KEY",
            },
        },
    }
    assert _extract_endpoint_from_cli_toml(raw)["api_key"] == "VLLM_CONFIG_KEY"

    del raw["api"]["vllm"]["api_key"]
    assert _extract_endpoint_from_cli_toml(raw)["api_key"] is None


def test_vllm_legacy_url_beats_environment(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "https://env.example/v1")
    raw = {
        "general": {"model": "custom-model"},
        "api": {"openai": {"base_url": "https://legacy.example/v1"}},
    }
    assert _extract_endpoint_from_cli_toml(raw)["base_url"] == (
        "https://legacy.example/v1"
    )


def test_flattened_empty_vllm_section_preserves_canonical_presence(monkeypatch):
    from chemgraph.utils.config_utils import (
        flatten_config,
        get_base_url_for_model_from_flat_config as flat,
    )

    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    config = flatten_config(
        {
            "api": {
                "openai": {"base_url": "https://api.openai.com/v1"},
                "vllm": {},
            }
        }
    )
    assert config["api_vllm_base_url"] == ""
    assert flat("custom-model", config) is None


def test_absent_vllm_section_uses_warned_legacy_fallback(
    monkeypatch, caplog, _propagate_logs
):
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    raw = {
        "general": {"model": "custom-model"},
        "api": {"openai": {"base_url": "https://legacy.example/v1"}},
    }
    with caplog.at_level(logging.WARNING):
        resolved = _extract_endpoint_from_cli_toml(raw)
    assert resolved["base_url"] == "https://legacy.example/v1"
    assert any("legacy [api.openai]" in r.message for r in caplog.records)


def test_ui_loader_preserves_absent_canonical_sections(tmp_path):
    from chemgraph.utils.config_utils import (
        get_base_url_for_model_from_nested_config as nested,
    )
    from ui.config import load_config

    cfg = tmp_path / "legacy.toml"
    cfg.write_text(
        '[general]\nmodel = "custom-model"\n'
        '[api.openai]\nbase_url = "https://legacy.example/v1"\n'
    )
    loaded = load_config(str(cfg))
    assert "argo" not in loaded["api"]
    assert "vllm" not in loaded["api"]
    assert nested("custom-model", loaded) == "https://legacy.example/v1"


def test_ui_editor_does_not_materialize_empty_vllm_section():
    from ui._pages.configuration import _update_vllm_config

    api = {"openai": {"base_url": "https://legacy.example/v1"}}
    _update_vllm_config(api, "")
    assert "vllm" not in api

    _update_vllm_config(api, "https://canonical.example/v1")
    assert api["vllm"]["base_url"] == "https://canonical.example/v1"


# ---------------------------------------------------------------------------
# check_api_keys resolves through endpoint CredentialPolicy
# ---------------------------------------------------------------------------


@pytest.fixture
def _clear_keys(monkeypatch):
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "ALCF_ACCESS_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def test_check_api_keys_missing_anthropic(_clear_keys):
    from chemgraph.cli.commands import check_api_keys

    ok, message = check_api_keys(ANTHROPIC_MODEL)
    assert ok is False
    assert "ANTHROPIC_API_KEY" in message


def test_check_api_keys_present_anthropic(_clear_keys, monkeypatch):
    from chemgraph.cli.commands import check_api_keys

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert check_api_keys(ANTHROPIC_MODEL) == (True, "")


def test_check_api_keys_argo_needs_no_key(_clear_keys):
    from chemgraph.cli.commands import check_api_keys

    assert check_api_keys("argo:gpt-4o") == (True, "")


def test_check_api_keys_codex_needs_no_key(_clear_keys):
    from chemgraph.cli.commands import check_api_keys

    assert check_api_keys("codex:gpt-5") == (True, "")


def test_check_api_keys_custom_model_uses_resolved_vllm_policy(_clear_keys):
    from chemgraph.cli.commands import check_api_keys

    assert check_api_keys(
        "custom-model",
        base_url="https://vllm.example/v1",
    ) == (True, "")


def test_vllm_api_key_precedence(monkeypatch):
    from chemgraph.models.endpoints import ModelRequest
    from chemgraph.models.endpoints.vllm import resolve_api_key

    monkeypatch.setenv("VLLM_API_KEY", "vllm-env")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    assert resolve_api_key(
        ModelRequest(model="custom", api_key="explicit")
    ) == "explicit"
    assert resolve_api_key(ModelRequest(model="custom")) == "vllm-env"


def test_vllm_openai_key_fallback_warns(monkeypatch, caplog, _propagate_logs):
    from chemgraph.models.endpoints import ModelRequest
    from chemgraph.models.endpoints.vllm import resolve_api_key

    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-env")
    with caplog.at_level(logging.WARNING):
        assert resolve_api_key(ModelRequest(model="custom")) == "openai-env"
    assert any("deprecated OPENAI_API_KEY" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# config_utils Argo-user helpers read the canonical [api.argo].user (#4)
# ---------------------------------------------------------------------------


def test_argo_user_nested_reads_canonical_section():
    from chemgraph.utils.config_utils import get_argo_user_from_nested_config

    cfg = {"api": {"argo": {"argo_user": "alice"}}}
    assert get_argo_user_from_nested_config(cfg) == "alice"


def test_argo_user_nested_legacy_fallback():
    from chemgraph.utils.config_utils import get_argo_user_from_nested_config

    cfg = {"api": {"openai": {"argo_user": "bob"}}}
    assert get_argo_user_from_nested_config(cfg) == "bob"


def test_argo_user_nested_canonical_beats_legacy():
    from chemgraph.utils.config_utils import get_argo_user_from_nested_config

    cfg = {
        "api": {
            "argo": {"argo_user": "alice"},
            "openai": {"argo_user": "bob"},
        }
    }
    assert get_argo_user_from_nested_config(cfg) == "alice"


def test_argo_user_flat_reads_canonical_key():
    from chemgraph.utils.config_utils import get_argo_user_from_flat_config

    assert (
        get_argo_user_from_flat_config({"api_argo_argo_user": "alice"})
        == "alice"
    )
    assert (
        get_argo_user_from_flat_config({"api_openai_argo_user": "bob"}) == "bob"
    )


def test_argo_user_compatibility_alias_still_works():
    from chemgraph.utils.config_utils import get_argo_user_from_nested_config

    assert get_argo_user_from_nested_config(
        {"api": {"argo": {"user": "alice"}}}
    ) == "alice"


# ---------------------------------------------------------------------------
# UI model options prioritize Argo based on [api.argo] configuration (#4)
# ---------------------------------------------------------------------------


def test_model_options_prioritize_argo_from_canonical_section():
    from chemgraph.utils.config_utils import get_model_options_for_nested_config
    from chemgraph.models.supported_models import supported_argo_models

    cfg = {"api": {"argo": {"base_url": "https://apps.inside.anl.gov/argoapi/v1"}}}
    assert (
        get_model_options_for_nested_config(cfg)[0] == supported_argo_models[0]
    )


def test_model_options_prioritize_argo_from_legacy_openai():
    from chemgraph.utils.config_utils import get_model_options_for_nested_config
    from chemgraph.models.supported_models import supported_argo_models

    cfg = {
        "api": {
            "openai": {
                "base_url": "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"
            }
        }
    }
    assert (
        get_model_options_for_nested_config(cfg)[0] == supported_argo_models[0]
    )


def test_model_options_no_argo_when_openai_is_direct():
    from chemgraph.utils.config_utils import get_model_options_for_nested_config
    from chemgraph.models.supported_models import supported_argo_models

    cfg = {"api": {"openai": {"base_url": "https://api.openai.com/v1"}}}
    assert (
        get_model_options_for_nested_config(cfg)[0] not in supported_argo_models
    )


# ---------------------------------------------------------------------------
# Direct OpenAI vs Argo route to distinct base URLs (#2)
# ---------------------------------------------------------------------------


def test_direct_openai_and_argo_resolve_distinct_urls():
    from chemgraph.utils.config_utils import (
        get_base_url_for_model_from_nested_config as nested,
    )

    cfg = {
        "api": {
            "openai": {"base_url": "https://api.openai.com/v1"},
            "argo": {"base_url": "https://apps.inside.anl.gov/argoapi/v1"},
        }
    }
    assert nested("gpt-4o", cfg) == "https://api.openai.com/v1"
    assert nested("argo:gpt-4o", cfg) == "https://apps.inside.anl.gov/argoapi/v1"


def test_repository_config_has_canonical_argo_and_vllm_sections():
    import tomllib
    from pathlib import Path

    config = tomllib.loads(Path("config.toml").read_text())
    assert config["api"]["openai"]["base_url"] == "https://api.openai.com/v1"
    assert "argo" in config["api"]
    assert "vllm" in config["api"]
