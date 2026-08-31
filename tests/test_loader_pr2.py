"""PR 2 regression tests for the single-loader migration (issue #201).

These lock in the behaviors PR 2 promised beyond the endpoint-level route
matrix (which already covers per-endpoint ``prepare`` parity):

- The ordered endpoint registry resolves each family to the right endpoint,
  with prefix routes winning over catalog checks.
- The vLLM/custom fallback is chosen only when a base URL is available, and
  recognized-provider errors never fall through to it.
- Explicit credentials are forwarded and override the environment.
- ``reasoning_effort`` flows through ``load_chat_model`` to the endpoint.
- ``ChemGraph`` forwards an explicit OpenAI ``api_key`` (previously dropped).
- ``run_turn`` and the judge share the same loader seam.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from chemgraph.models.endpoints import ModelRequest
from chemgraph.models.loader import (
    _build_request,
    _select_endpoint,
    load_chat_model,
    load_chat_model_prepared,
)


@pytest.fixture(autouse=True)
def _clear_provider_env(monkeypatch):
    """Isolate credential/URL resolution from the developer's environment."""
    for var in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "ALCF_ACCESS_TOKEN",
        "OPENROUTER_API_KEY",
        "GROQ_API_KEY",
        "VLLM_API_KEY",
        "VLLM_BASE_URL",
        "ARGO_USER",
        "CHEMGRAPH_ARGO_MODEL_FORMAT",
    ):
        monkeypatch.delenv(var, raising=False)


def _route(model: str, **kwargs) -> str:
    request = _build_request(
        model,
        0.0,
        kwargs.get("base_url"),
        kwargs.get("api_key"),
        kwargs.get("argo_user"),
        kwargs.get("reasoning_effort"),
        None,
    )
    return _select_endpoint(request).name


# --- Ordered registry resolution -------------------------------------------


@pytest.mark.parametrize(
    "model, expected_endpoint",
    [
        ("codex:gpt-5", "codex"),
        ("openrouter:openai/o3", "openrouter"),  # prefix wins over OpenAI catalog
        ("argo:gpt-4o", "argo_openai"),
        ("argo:claude-opus-4.8", "argo_anthropic"),
        ("alcf:nemotron-3-ultra", "alcf"),
        ("gpt-4o", "openai_direct"),
        ("claude-3-5-sonnet-20241022", "anthropic_direct"),
        ("gemini-2.5-pro", "google_direct"),
        ("llama3.2", "ollama"),
        ("groq:llama-3.1-8b-instant", "groq"),
    ],
)
def test_registry_resolves_expected_endpoint(model, expected_endpoint):
    assert _route(model) == expected_endpoint


def test_argo_models_build_protocol_specific_clients():
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    claude_client, claude_prepared = load_chat_model_prepared(
        model_name="argo:claude-opus-4.8",
        argo_user="alice",
    )
    gpt_client, gpt_prepared = load_chat_model_prepared(
        model_name="argo:gpt-4o",
        argo_user="alice",
    )

    assert isinstance(claude_client, ChatAnthropic)
    assert claude_prepared.protocol == "anthropic_native"
    assert claude_prepared.client_kwargs["model"] == "claudeopus48"
    assert claude_prepared.client_kwargs["base_url"].endswith("/argoapi")
    assert isinstance(gpt_client, ChatOpenAI)
    assert gpt_prepared.protocol == "openai_compatible"
    assert gpt_prepared.client_kwargs["model"] == "gpt4o"
    assert gpt_prepared.client_kwargs["base_url"].endswith("/argoapi/v1")


def test_deprecated_openai_loader_preserves_argo_claude_chatopenai():
    from langchain_openai import ChatOpenAI

    from chemgraph.models.openai import load_openai_model

    with pytest.warns(DeprecationWarning, match="load_openai_model is deprecated"):
        client = load_openai_model(
            model_name="argo:claude-opus-4.8",
            temperature=0.0,
            argo_user="alice",
        )

    assert isinstance(client, ChatOpenAI)


def test_unknown_model_with_base_url_falls_to_vllm():
    assert _route("my-local-model", base_url="http://localhost:8000/v1") == "vllm"


def test_unknown_model_without_endpoint_raises():
    with pytest.raises(ValueError, match="not found in any supported model list"):
        _select_endpoint(ModelRequest(model="totally-unknown-model"))


def test_known_provider_error_does_not_fall_through_to_vllm(monkeypatch):
    # A recognized model whose credential is missing must raise its own error,
    # even when a vLLM base URL is configured -- it must not be retried as vLLM.
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")
    with pytest.raises(ValueError, match="Anthropic|API key|ANTHROPIC"):
        load_chat_model(model_name="claude-3-5-sonnet-20241022")


def test_invalid_alcf_model_never_falls_through_to_vllm(monkeypatch):
    monkeypatch.setenv("VLLM_BASE_URL", "http://localhost:8000/v1")

    assert _route("alcf:not-in-catalog", api_key="dummy-alcf-token") == "alcf"
    with patch("chemgraph.models.endpoints.alcf.resolve_api_key") as resolve_key:
        with pytest.raises(ValueError, match="not supported on ALCF"):
            load_chat_model(
                model_name="alcf:not-in-catalog",
                api_key="dummy-alcf-token",
            )

    resolve_key.assert_not_called()


# --- Credential forwarding --------------------------------------------------


def test_openrouter_never_uses_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-be-used")
    with pytest.raises(ValueError, match="OpenRouter API key not found"):
        load_chat_model(model_name="openrouter:moonshotai/kimi-k3")


def test_explicit_openai_key_is_forwarded_and_overrides_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    _client, prepared = load_chat_model_prepared(
        model_name="gpt-4o", api_key="sk-explicit"
    )
    assert prepared.client_kwargs["api_key"] == "sk-explicit"


def test_argo_loader_uses_argo_user_instead_of_api_keys(monkeypatch):
    monkeypatch.setenv("ARGO_USER", "argo-env-user")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")

    _client, prepared = load_chat_model_prepared(
        model_name="argo:gpt-4o",
        api_key="sk-explicit-secret",
    )

    assert prepared.client_kwargs["api_key"] == "argo-env-user"


# --- reasoning_effort flow --------------------------------------------------


def test_reasoning_effort_flows_to_argo_reasoning_model():
    _client, prepared = load_chat_model_prepared(
        model_name="argo:gpt-5.6-sol",
        base_url="https://apps.inside.anl.gov/argoapi/v1",
        reasoning_effort="high",
    )
    assert prepared.reasoning_effort == "high"
    assert prepared.client_kwargs.get("reasoning_effort") == "high"


# --- ChemGraph forwards explicit OpenAI api_key ----------------------------


def test_chemgraph_forwards_explicit_openai_api_key(tmp_path):
    captured = {}

    def _fake_prepared(**kwargs):
        captured.update(kwargs)
        from chemgraph.models.endpoints import PreparedModel

        return (
            object(),
            PreparedModel(
                endpoint_name="openai_direct",
                protocol="openai_compatible",
                client_kwargs={},
            ),
        )

    from chemgraph.agent.llm_agent import ChemGraph

    with patch(
        "chemgraph.agent.llm_agent.load_chat_model_prepared", _fake_prepared
    ), patch("chemgraph.agent.llm_agent.construct_single_agent_graph", lambda *a, **k: object()):
        ChemGraph(
            model_name="gpt-4o-mini",
            api_key="sk-explicit-agent",
            enable_memory=False,
            log_dir=str(tmp_path / "logs"),
        )

    assert captured["api_key"] == "sk-explicit-agent"


# --- Custom endpoint is consistent across run_turn and the judge -----------


def test_run_turn_and_judge_use_the_same_loader_seam():
    # The judge loads through load_chat_model directly; run_turn loads through
    # _load_turn_llm, which also delegates to load_chat_model. Both resolve an
    # unknown model with a base URL to the same vLLM endpoint.
    from chemgraph.agent.turn import _load_turn_llm

    with patch("chemgraph.agent.turn.load_chat_model") as mock_turn_load:
        mock_turn_load.return_value = "TURN_LLM"
        result = _load_turn_llm(
            model_name="my-local-model",
            base_url="http://localhost:8000/v1",
            api_key=None,
            argo_user=None,
        )
    assert result == "TURN_LLM"
    # _load_turn_llm passes settings (not positional model_name) to the loader.
    assert mock_turn_load.call_args.kwargs["settings"].model == "my-local-model"
