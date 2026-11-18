"""Offline configuration, credential isolation, and endpoint client contracts."""

import json
import os
import sys
from dataclasses import replace

import pytest

from chemgraph.api.settings import ConfigurationError, Provider, Settings
from chemgraph.api.providers import credentials, model_status, provenance
from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints.registry import select_endpoint
from chemgraph.models.supported_models import (
    supported_alcf_models,
    supported_ollama_models,
    supported_anthropic_models,
    supported_gemini_models,
)


@pytest.fixture(autouse=True)
def isolate_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith("CHEMGRAPH_WEB_") or name in {
            "ARGO_USER",
            "ALCF_ACCESS_TOKEN",
            "OPENAI_API_KEY",
            "VLLM_API_KEY",
            "VLLM_BASE_URL",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
            "OPENROUTER_API_KEY",
            "REVIEW_KEY",
        }:
            monkeypatch.delenv(name)


def test_provider_file_and_complete_json_override(tmp_path, monkeypatch):
    path = tmp_path / "providers.toml"
    path.write_text('[providers.Argo]\nmodel="argo:gpt-4o"\nargo_user="service-user"\n')
    monkeypatch.setenv("CHEMGRAPH_WEB_PROVIDERS_FILE", str(path))
    assert list(Settings.from_env().providers) == ["Argo"]
    monkeypatch.setenv("CHEMGRAPH_WEB_PROVIDERS", "{}")
    path.write_text("invalid TOML ignored because JSON overrides it")
    assert Settings.from_env().providers == {}
    monkeypatch.delenv("CHEMGRAPH_WEB_PROVIDERS")
    with pytest.raises(ConfigurationError):
        Settings.from_env()


@pytest.mark.parametrize(
    "value",
    [
        "",
        "[]",
        "null",
        '{"Lab":{"model":"argo:gpt-4o","api_key":"literal-secret-marker"}}',
        '{" ":{"model":"argo:gpt-4o"}}',
        '{"Lab":{"model":"argo:gpt-4o","base_url":"https://user:literal-secret-marker@example.com"}}',
    ],
)
def test_invalid_configuration_never_echoes_values(monkeypatch, value):
    monkeypatch.setenv("CHEMGRAPH_WEB_PROVIDERS", value)
    with pytest.raises(ConfigurationError) as exc:
        Settings.from_env()
    assert "literal-secret-marker" not in str(exc.value)


@pytest.mark.parametrize(
    "model", ["argo:unknown-model", "alcf:unknown-model", "codex:gpt-5", "groq:"]
)
def test_invalid_routes_fail_before_serving(model):
    with pytest.raises(ConfigurationError):
        model_status(Settings(providers={"Lab": Provider(model=model)}))


def test_shared_identity_and_alcf_override(monkeypatch):
    argo = Provider(model="argo:gpt-4o")
    assert not model_status(Settings(providers={"Argo": argo}))["Argo"]["configured"]
    monkeypatch.setenv("ARGO_USER", "shared-user")
    monkeypatch.setenv("OPENAI_API_KEY", "never-forward-this")
    assert credentials(argo).argo_user == "shared-user"
    assert credentials(argo).api_key is None
    assert (
        credentials(argo.model_copy(update={"argo_user": "explicit-user"})).argo_user
        == "explicit-user"
    )
    alcf = Provider(model=supported_alcf_models[0])
    monkeypatch.setenv("ALCF_ACCESS_TOKEN", "alcf-token")
    assert credentials(alcf).api_key == "alcf-token"
    overridden = alcf.model_copy(update={"api_key_env": "REVIEW_KEY"})
    with pytest.raises(ValueError, match="missing_credentials"):
        credentials(overridden)
    monkeypatch.setenv("REVIEW_KEY", "override-token")
    assert credentials(overridden).api_key == "override-token"


@pytest.mark.parametrize(
    "model,variable",
    [
        ("gpt-4o", "OPENAI_API_KEY"),
        (supported_anthropic_models[0], "ANTHROPIC_API_KEY"),
        (supported_gemini_models[0], "GEMINI_API_KEY"),
        ("groq:test", "GROQ_API_KEY"),
        ("openrouter:test/model", "OPENROUTER_API_KEY"),
    ],
)
def test_provider_specific_defaults(model, variable, monkeypatch):
    provider = Provider(model=model)
    assert not model_status(Settings(providers={"Model": provider}))["Model"][
        "configured"
    ]
    monkeypatch.setenv(variable, "test-token")
    assert credentials(provider).api_key == "test-token"


def test_custom_endpoint_never_inherits_hosted_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "hosted-key")
    provider = Provider(model="test-model", base_url="http://localhost:9999/v1")
    resolved = credentials(provider)
    assert resolved.api_key != "hosted-key"
    prepared = select_endpoint(
        ModelRequest(model=provider.model, base_url=provider.base_url)
    ).prepare_request(
        ModelRequest(
            model=provider.model, base_url=provider.base_url, api_key=resolved.api_key
        )
    )
    assert prepared.client_kwargs["api_key"] == resolved.api_key


def test_provenance_ignores_credential_rotation_but_tracks_endpoint(monkeypatch):
    settings = Settings(
        providers={"Lab": Provider(model="argo:gpt-4o", argo_user="shared")}
    )
    before = provenance(settings, "Lab")
    settings.providers["Lab"].argo_user = "rotated"
    assert provenance(settings, "Lab") == before
    settings.providers["Lab"].base_url = "https://lab.example/v1"
    assert provenance(settings, "Lab") != before


@pytest.mark.parametrize(
    "configured,expected",
    [
        ({}, 1),
        ({"Lab": {"model": "argo:gpt-4o"}}, 1),
        ({"Lab": {"model": "argo:gpt-4o", "argo_user": "shared"}}, 0),
        ({"Lab": {"model": "alcf:bad"}}, 2),
    ],
)
def test_check_config_is_noninteractive_and_creates_no_storage(
    configured, expected, monkeypatch, tmp_path, capsys
):
    from chemgraph.api.__main__ import main
    import socket

    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *a, **k: pytest.fail("Configuration contacted the network"),
    )
    monkeypatch.setattr(
        "getpass.getpass",
        lambda *a, **k: pytest.fail("Configuration prompted for a credential"),
    )
    monkeypatch.setattr(sys, "argv", ["chemgraph.api", "--check-config"])
    monkeypatch.setenv("CHEMGRAPH_WEB_PROVIDERS", json.dumps(configured))
    monkeypatch.setenv("CHEMGRAPH_WEB_DATA_DIR", str(tmp_path / "unused"))
    before = dict(os.environ)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == expected
    assert dict(os.environ) == before
    assert not (tmp_path / "unused").exists()
    output = capsys.readouterr()
    if expected != 2:
        assert json.loads(output.out)["connectivity_verified"] is False


@pytest.mark.parametrize(
    "model,module,client_name",
    [
        (supported_ollama_models[0], "chemgraph.models.local_model", "ChatOllama"),
        (
            supported_anthropic_models[0],
            "chemgraph.models.protocols.anthropic_native",
            "ChatAnthropic",
        ),
        (
            supported_gemini_models[0],
            "chemgraph.models.protocols.google_native",
            "ChatGoogleGenerativeAI",
        ),
    ],
)
def test_custom_urls_reach_final_clients(model, module, client_name, monkeypatch):
    from chemgraph.models.loader import load_chat_model_prepared

    captured = []
    monkeypatch.setattr(
        f"{module}.{client_name}", lambda **kwargs: captured.append(kwargs) or object()
    )
    load_chat_model_prepared(
        model, api_key="test-token", base_url="https://provider.example", timeout_s=17
    )
    assert captured[0]["base_url"] == "https://provider.example"
    if client_name == "ChatOllama":
        assert captured[0]["client_kwargs"]["timeout"] == 17
    else:
        assert captured[0]["timeout"] == 17


def test_check_config_does_not_construct_clients(monkeypatch):
    from chemgraph.api import providers
    from chemgraph.models.endpoints import registry

    original = registry.select_endpoint
    monkeypatch.setattr(
        registry,
        "select_endpoint",
        lambda request: replace(
            original(request),
            protocol_build=lambda **kw: pytest.fail("Client constructed"),
        ),
    )
    assert providers.model_status(
        Settings(providers={"Argo": Provider(model="argo:gpt-4o", argo_user="shared")})
    )["Argo"]["configured"]


@pytest.mark.parametrize(
    "exception,expected",
    [
        (TimeoutError("private text"), "provider_timeout"),
        (ConnectionError("private text"), "provider_connection"),
    ],
)
def test_safe_transport_errors(exception, expected):
    from chemgraph.api.errors import public_error

    outer = RuntimeError("raw provider response")
    outer.__cause__ = exception
    code, message = public_error(outer)
    assert code == expected
    assert "private text" not in message and "raw provider response" not in message
