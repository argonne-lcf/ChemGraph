"""Tests for provider readiness detection and ALCF token handling."""

import json

import pytest

from ui import alcf_auth, providers


@pytest.fixture()
def clean_env(monkeypatch, tmp_path):
    """Remove provider credentials and isolate token caches."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
        "OPENROUTER_API_KEY",
        "ARGO_USER",
        "ALCF_ACCESS_TOKEN",
    ):
        # setenv-then-delenv registers an undo even for vars absent before
        # the test, so values exported by the code under test (e.g.
        # ensure_access_token) cannot leak into later tests.
        monkeypatch.setenv(var, "sentinel")
        monkeypatch.delenv(var)
    monkeypatch.setattr(
        alcf_auth, "CHEMGRAPH_TOKENS_PATH", str(tmp_path / "cg_tokens.json")
    )
    monkeypatch.setattr(
        alcf_auth, "HELPER_TOKENS_PATH", str(tmp_path / "helper_tokens.json")
    )
    return tmp_path


def _config(argo_user=""):
    return {
        "api": {
            "openai": {"base_url": providers.OPENAI_DEFAULT_BASE_URL},
            "argo": {
                "base_url": providers.ARGO_DEFAULT_BASE_URL,
                "argo_user": argo_user,
            },
        }
    }


def test_no_provider_ready_without_credentials(clean_env):
    assert providers.any_provider_ready(_config()) is False


def test_argo_ready_with_config_username(clean_env):
    config = _config(argo_user="aturing")

    status = providers.provider_status(
        providers.get_provider(providers.ARGO), config
    )

    assert status.ready is True
    assert "aturing" in status.detail
    assert providers.any_provider_ready(config) is True


def test_argo_ready_with_env_username(clean_env, monkeypatch):
    monkeypatch.setenv("ARGO_USER", "aturing")

    status = providers.provider_status(
        providers.get_provider(providers.ARGO), _config()
    )

    assert status.ready is True


def test_api_key_provider_ready_from_env(clean_env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

    statuses = {
        s.info.id: s.ready for s in providers.all_provider_statuses(_config())
    }

    assert statuses[providers.ANTHROPIC] is True
    assert statuses[providers.OPENAI] is False


def test_local_provider_does_not_satisfy_first_run_check(clean_env):
    statuses = {
        s.info.id: s.ready for s in providers.all_provider_statuses(_config())
    }

    assert statuses[providers.OLLAMA] is True  # nominally ready...
    assert providers.any_provider_ready(_config()) is False  # ...but excluded


def test_provider_for_model_maps_prefixes_and_lists():
    assert providers.provider_for_model("argo:gpt-4o").id == providers.ARGO
    assert providers.provider_for_model("alcf:foo/Bar").id == providers.ALCF
    assert providers.provider_for_model("groq:x").id == providers.GROQ
    assert (
        providers.provider_for_model("openrouter:a/b").id == providers.OPENROUTER
    )
    assert providers.provider_for_model("gpt-4o-mini").id == providers.OPENAI
    assert (
        providers.provider_for_model("gemini-2.5-flash").id == providers.GOOGLE
    )
    assert providers.provider_for_model("llama3.2").id == providers.OLLAMA
    # Unknown names use the configured OpenAI-compatible vLLM endpoint.
    assert providers.provider_for_model("mystery-model").id == providers.VLLM


def test_vllm_ready_only_with_endpoint_url(clean_env):
    info = providers.get_provider(providers.VLLM)
    config = _config()

    assert providers.provider_status(info, config).ready is False

    config["api"]["vllm"] = {"base_url": "https://vllm.example/v1"}
    assert providers.provider_status(info, config).ready is True


def test_every_provider_default_model_maps_back_to_it():
    for info in providers.PROVIDERS:
        mapped = providers.provider_for_model(info.default_model)
        assert mapped is not None and mapped.id == info.id, info.id


# ---------------------------------------------------------------------------
# ALCF token cache handling
# ---------------------------------------------------------------------------


def test_alcf_status_env_token_wins(clean_env, monkeypatch):
    monkeypatch.setenv("ALCF_ACCESS_TOKEN", "tok")

    assert alcf_auth.token_status()["state"] == "env"
    assert alcf_auth.ensure_access_token() == "tok"


def test_alcf_status_logged_out_without_caches(clean_env):
    assert alcf_auth.token_status()["state"] == "logged_out"
    assert alcf_auth.ensure_access_token(allow_refresh=False) is None


def test_alcf_valid_cached_record_is_exported(clean_env, monkeypatch):
    record = {
        "access_token": "cached-tok",
        "refresh_token": "r",
        "expires_at_seconds": 4102444800,  # far future
    }
    alcf_auth.save_token_record(record)

    assert alcf_auth.token_status()["state"] == "valid"
    assert alcf_auth.ensure_access_token(allow_refresh=False) == "cached-tok"
    import os

    assert os.environ["ALCF_ACCESS_TOKEN"] == "cached-tok"


def test_alcf_expired_record_reports_refreshable(clean_env):
    alcf_auth.save_token_record(
        {"access_token": "old", "refresh_token": "r", "expires_at_seconds": 1}
    )

    assert alcf_auth.token_status()["state"] == "refreshable"
    assert alcf_auth.ensure_access_token(allow_refresh=False) is None


def test_alcf_reads_helper_script_nested_format(clean_env, tmp_path):
    helper = {
        "data": {
            "DEFAULT": {
                alcf_auth.GATEWAY_CLIENT_ID: {
                    "access_token": "helper-tok",
                    "refresh_token": "r",
                    "expires_at_seconds": 4102444800,
                }
            }
        }
    }
    with open(alcf_auth.HELPER_TOKENS_PATH, "w") as f:
        json.dump(helper, f)

    record, source = alcf_auth.read_token_record()

    assert record["access_token"] == "helper-tok"
    assert source == alcf_auth.HELPER_TOKENS_PATH
    assert alcf_auth.token_status()["state"] == "valid"


def test_alcf_refresh_uses_refresh_token(clean_env, monkeypatch):
    alcf_auth.save_token_record(
        {"access_token": "old", "refresh_token": "r", "expires_at_seconds": 1}
    )
    monkeypatch.setattr(
        alcf_auth,
        "_refresh_record",
        lambda record: {
            "access_token": "fresh",
            "refresh_token": record["refresh_token"],
            "expires_at_seconds": 4102444800,
        },
    )

    assert alcf_auth.ensure_access_token() == "fresh"
    # The refreshed record is persisted for next time.
    record, _ = alcf_auth.read_token_record()
    assert record["access_token"] == "fresh"


def test_alcf_exported_env_token_keeps_expiry_semantics(clean_env, monkeypatch):
    # Simulate a completed in-UI login whose access token later expired:
    # the env var holds the same (stale) token the cache holds.
    alcf_auth.save_token_record(
        {"access_token": "stale", "refresh_token": "r", "expires_at_seconds": 1}
    )
    monkeypatch.setenv("ALCF_ACCESS_TOKEN", "stale")

    # Status must not blindly trust the env copy of our own cached token.
    assert alcf_auth.token_status()["state"] == "refreshable"

    monkeypatch.setattr(
        alcf_auth,
        "_refresh_record",
        lambda record: {
            "access_token": "fresh",
            "refresh_token": "r",
            "expires_at_seconds": 4102444800,
        },
    )
    assert alcf_auth.ensure_access_token() == "fresh"
    import os

    assert os.environ["ALCF_ACCESS_TOKEN"] == "fresh"


def test_alcf_external_env_token_is_trusted(clean_env, monkeypatch):
    monkeypatch.setenv("ALCF_ACCESS_TOKEN", "external")
    alcf_auth.save_token_record(
        {"access_token": "cached", "refresh_token": "r", "expires_at_seconds": 1}
    )

    # A token that does not match the cache was supplied by the user.
    assert alcf_auth.token_status()["state"] == "env"
    assert alcf_auth.ensure_access_token(allow_refresh=False) == "external"


def test_align_base_url_for_provider_keeps_separate_sections():
    config = {"api": {"openai": {"base_url": "https://api.openai.com/v1"}}}

    providers.align_base_url_for_provider(config, providers.ARGO)
    assert config["api"]["argo"]["base_url"] == providers.ARGO_DEFAULT_BASE_URL
    assert config["api"]["openai"]["base_url"] == providers.OPENAI_DEFAULT_BASE_URL

    providers.align_base_url_for_provider(config, providers.OPENAI)
    assert config["api"]["openai"]["base_url"] == providers.OPENAI_DEFAULT_BASE_URL

    # Other providers leave both sections alone.
    providers.align_base_url_for_provider(config, providers.ANTHROPIC)
    assert config["api"]["argo"]["base_url"] == providers.ARGO_DEFAULT_BASE_URL
    assert config["api"]["openai"]["base_url"] == providers.OPENAI_DEFAULT_BASE_URL


def test_alcf_logout_clears_cache_and_env(clean_env, monkeypatch):
    monkeypatch.setenv("ALCF_ACCESS_TOKEN", "tok")
    alcf_auth.save_token_record(
        {"access_token": "tok", "expires_at_seconds": 4102444800}
    )

    alcf_auth.logout()

    import os

    assert "ALCF_ACCESS_TOKEN" not in os.environ
    assert alcf_auth.read_token_record() == (None, None)
