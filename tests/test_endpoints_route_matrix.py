"""Route-matrix tests for the endpoint × protocol layer.

Each case exercises an endpoint's ``prepare`` directly and asserts the resolved
endpoint name, protocol, base URL, wire model name, credential source, and
final client kwargs, including protocol-specific Argo routing.
"""

from __future__ import annotations

import pytest

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.endpoints import alcf as alcf_ep
from chemgraph.models.endpoints import anthropic_direct as anthropic_ep
from chemgraph.models.endpoints import argo as argo_ep
from chemgraph.models.endpoints import google_direct as google_ep
from chemgraph.models.endpoints import openai_direct as openai_ep
from chemgraph.models.endpoints import openrouter as openrouter_ep
from chemgraph.models.endpoints import aurora as aurora_ep
from chemgraph.models.endpoints import vllm as vllm_ep
from chemgraph.models.supported_models import (
    ALCF_DEFAULT_BASE_URL,
    ALCF_METIS_BASE_URL,
    ALCF_MINERVA_BASE_URL,
    ARGO_DEFAULT_ANTHROPIC_BASE_URL,
    ARGO_DEFAULT_BASE_URL,
    AURORA_DEFAULT_BASE_URL,
    OPENROUTER_DEFAULT_BASE_URL,
)

ARGO_HOSTED_URL = "https://apps.inside.anl.gov/argoapi/v1"


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Isolate credential/URL resolution from the developer's environment."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "ALCF_ACCESS_TOKEN",
        "OPENROUTER_API_KEY",
        "AURORA_API_KEY",
        "AURORA_BASE_URL",
        "VLLM_API_KEY",
        "VLLM_BASE_URL",
        "ARGO_USER",
        "CHEMGRAPH_ARGO_MODEL_FORMAT",
    ):
        monkeypatch.delenv(var, raising=False)


# --- Direct OpenAI ---------------------------------------------------------


def test_openai_direct_route():
    prepared = openai_ep.prepare(
        ModelRequest(model="gpt-4o", temperature=0.0, api_key="sk-explicit")
    )
    assert prepared.endpoint_name == "openai_direct"
    assert prepared.protocol == "openai_compatible"
    assert prepared.supports_structured_output is True
    kwargs = prepared.client_kwargs
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["api_key"] == "sk-explicit"
    assert "base_url" not in kwargs  # plain OpenAI shape
    assert kwargs["max_tokens"] == 6000
    assert kwargs["temperature"] == 0.0


def test_openai_direct_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        openai_ep.prepare(ModelRequest(model="not-a-real-model", api_key="k"))


# --- Argo hosted (wire names) ----------------------------------------------


def test_argo_hosted_route_uses_wire_name_and_user_payload():
    prepared = argo_ep.prepare_openai(
        ModelRequest(model="argo:gpt-4o", base_url=ARGO_HOSTED_URL, argo_user="alice")
    )
    assert prepared.endpoint_name == "argo_openai"
    assert prepared.protocol == "openai_compatible"
    assert prepared.supports_structured_output is False
    kwargs = prepared.client_kwargs
    assert kwargs["model"] == "gpt4o"  # wire name
    assert kwargs["base_url"] == ARGO_HOSTED_URL
    assert kwargs["model_kwargs"] == {"user": "alice"}
    assert kwargs["max_tokens"] == 4000


def test_argo_defaults_base_url_when_absent():
    prepared = argo_ep.prepare_openai(ModelRequest(model="argo:gpt-4o"))
    assert prepared.client_kwargs["base_url"] == ARGO_DEFAULT_BASE_URL


def test_argo_claude_uses_anthropic_protocol_and_wire_name():
    prepared = argo_ep.prepare_anthropic(
        ModelRequest(
            model="argo:claude-opus-4.8",
            base_url=ARGO_HOSTED_URL,
            argo_user="alice",
        )
    )
    assert prepared.endpoint_name == "argo_anthropic"
    assert prepared.protocol == "anthropic_native"
    assert prepared.supports_structured_output is False
    assert prepared.client_kwargs == {
        "model": "claudeopus48",
        "api_key": "alice",
        "base_url": ARGO_DEFAULT_ANTHROPIC_BASE_URL,
        "max_tokens": 4000,
        "streaming": True,
    }


@pytest.mark.parametrize(
    "configured_url, expected_url",
    [
        (ARGO_HOSTED_URL, ARGO_DEFAULT_ANTHROPIC_BASE_URL),
        (
            "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/",
            ARGO_DEFAULT_ANTHROPIC_BASE_URL,
        ),
        ("http://localhost:8080/v1", "http://localhost:8080"),
    ],
)
def test_argo_anthropic_normalizes_base_url(configured_url, expected_url):
    prepared = argo_ep.prepare_anthropic(
        ModelRequest(
            model="argo:claude-opus-4.8",
            base_url=configured_url,
            argo_user="alice",
        )
    )
    assert prepared.client_kwargs["base_url"] == expected_url


def test_argo_anthropic_never_uses_anthropic_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    prepared = argo_ep.prepare_anthropic(
        ModelRequest(model="argo:claude-opus-4.8", argo_user="alice")
    )
    assert prepared.client_kwargs["api_key"] == "alice"


def test_argo_minimal_parameter_model_omits_sampling():
    prepared = argo_ep.prepare_openai(
        ModelRequest(model="argo:gpt-5", base_url=ARGO_HOSTED_URL)
    )
    kwargs = prepared.client_kwargs
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs


# --- Argo compatible (shim/proxy, prefix stripped) -------------------------


def test_argo_compatible_route_strips_prefix_for_local_shim():
    prepared = argo_ep.prepare_openai(
        ModelRequest(model="argo:gpt-4.1-mini", base_url="http://localhost:8080/v1")
    )
    # A non-argoapi custom endpoint keeps the OpenAI-style name (prefix
    # stripped) and does not attach a hosted-Argo user payload.
    assert prepared.client_kwargs["model"] == "gpt-4.1-mini"
    assert "model_kwargs" not in prepared.client_kwargs


# --- ALCF three clusters ----------------------------------------------------


@pytest.mark.parametrize(
    "model, expected_url, expected_wire",
    [
        (
            "alcf:meta-llama/Llama-3.3-70B-Instruct",
            ALCF_DEFAULT_BASE_URL,
            "meta-llama/Llama-3.3-70B-Instruct",
        ),
        ("alcf:nemotron-3-ultra", ALCF_MINERVA_BASE_URL, "nemotron-3-ultra"),
        (
            "alcf:Mistral-Large-3-675B-Instruct-2512",
            ALCF_METIS_BASE_URL,
            "Mistral-Large-3-675B-Instruct-2512",
        ),
    ],
)
def test_alcf_cluster_routing(model, expected_url, expected_wire):
    prepared = alcf_ep.prepare(ModelRequest(model=model, api_key="tok"))
    assert prepared.endpoint_name == "alcf"
    assert prepared.client_kwargs["base_url"] == expected_url
    assert prepared.client_kwargs["model"] == expected_wire
    assert prepared.client_kwargs["api_key"] == "tok"


def test_alcf_requires_token():
    with pytest.raises(ValueError, match="ALCF access token not found"):
        alcf_ep.prepare(ModelRequest(model="alcf:nemotron-3-ultra"))


# --- OpenRouter -------------------------------------------------------------


def test_openrouter_curated_route():
    prepared = openrouter_ep.prepare(
        ModelRequest(model="openrouter:moonshotai/kimi-k3", api_key="or-key")
    )
    assert prepared.endpoint_name == "openrouter"
    kwargs = prepared.client_kwargs
    assert kwargs["model"] == "moonshotai/kimi-k3"  # prefix stripped, slug verbatim
    assert kwargs["base_url"] == OPENROUTER_DEFAULT_BASE_URL
    assert kwargs["max_tokens"] == 8000
    assert "top_p" not in kwargs  # restricted optional params


def test_openrouter_uncurated_slug_still_resolves():
    prepared = openrouter_ep.prepare(
        ModelRequest(model="openrouter:some/unknown-model", api_key="or-key")
    )
    assert prepared.client_kwargs["model"] == "some/unknown-model"


def test_openrouter_never_uses_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    with pytest.raises(ValueError, match="OpenRouter API key not found"):
        openrouter_ep.prepare(ModelRequest(model="openrouter:moonshotai/kimi-k3"))


# --- Anthropic / Gemini direct ---------------------------------------------


def test_anthropic_direct_route():
    prepared = anthropic_ep.prepare(
        ModelRequest(model="claude-3-5-sonnet-20241022", api_key="ak")
    )
    assert prepared.endpoint_name == "anthropic_direct"
    assert prepared.protocol == "anthropic_native"
    kwargs = prepared.client_kwargs
    assert kwargs["model"] == "claude-3-5-sonnet-20241022"
    assert kwargs["api_key"] == "ak"
    assert kwargs["max_tokens"] == 6000


def test_google_direct_route():
    prepared = google_ep.prepare(
        ModelRequest(model="gemini-2.5-pro", api_key="gk")
    )
    assert prepared.endpoint_name == "google_direct"
    assert prepared.protocol == "google_native"
    kwargs = prepared.client_kwargs
    assert kwargs["model"] == "gemini-2.5-pro"
    assert kwargs["max_output_tokens"] == 6000


# --- Aurora on-node ---------------------------------------------------------


def test_aurora_route_strips_prefix_and_uses_configured_base_url():
    prepared = aurora_ep.prepare(
        ModelRequest(
            model="aurora:gpt-oss-120b",
            base_url="http://x4000c0s0b0n0:8000/v1",
            api_key="dummy",
        )
    )
    assert prepared.endpoint_name == "aurora"
    assert prepared.protocol == "openai_compatible"
    kwargs = prepared.client_kwargs
    assert kwargs["model"] == "gpt-oss-120b"
    assert kwargs["base_url"] == "http://x4000c0s0b0n0:8000/v1"
    assert kwargs["api_key"] == "dummy"
    assert kwargs["temperature"] == 0.0


def test_aurora_route_falls_back_to_default_base_url_and_dummy_key():
    prepared = aurora_ep.prepare(ModelRequest(model="aurora:gpt-oss-120b"))
    kwargs = prepared.client_kwargs
    assert kwargs["base_url"] == AURORA_DEFAULT_BASE_URL
    assert kwargs["api_key"] == "dummy"  # placeholder when no key set


def test_aurora_route_env_base_url_used_when_no_explicit(monkeypatch):
    monkeypatch.setenv("AURORA_BASE_URL", "http://x4000c0s0b0n0:8000/v1")
    prepared = aurora_ep.prepare(ModelRequest(model="aurora:nemotron-3-ultra"))
    kwargs = prepared.client_kwargs
    assert kwargs["base_url"] == "http://x4000c0s0b0n0:8000/v1"
    assert kwargs["model"] == "nemotron-3-ultra"


def test_aurora_spec_matches_only_aurora_prefix():
    assert aurora_ep.SPEC.matches("aurora:gpt-oss-120b") is True
    assert aurora_ep.SPEC.matches("aurora:some-served-id") is True
    assert aurora_ep.SPEC.matches("openai:gpt-4o") is False
    assert aurora_ep.SPEC.matches("openrouter:x/y") is False


# --- vLLM / custom fallback -------------------------------------------------


def test_vllm_route_with_configured_base_url():
    prepared = vllm_ep.prepare(
        ModelRequest(model="my-local-model", base_url="http://localhost:8000/v1")
    )
    assert prepared.endpoint_name == "vllm"
    kwargs = prepared.client_kwargs
    assert kwargs["model"] == "my-local-model"
    assert kwargs["base_url"] == "http://localhost:8000/v1"
    assert kwargs["api_key"] == "dummy_vllm_key"  # placeholder when no key set


def test_vllm_without_endpoint_raises():
    with pytest.raises(ValueError, match="missing base URL"):
        vllm_ep.prepare(ModelRequest(model="my-local-model"))


def test_vllm_can_handle_reflects_configured_url(monkeypatch):
    assert vllm_ep.can_handle(ModelRequest(model="x")) is False
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    assert vllm_ep.can_handle(ModelRequest(model="x")) is True


# --- Credential precedence --------------------------------------------------


def test_explicit_key_overrides_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    prepared = openai_ep.prepare(ModelRequest(model="gpt-4o", api_key="sk-explicit"))
    assert prepared.client_kwargs["api_key"] == "sk-explicit"
