from __future__ import annotations

import pytest

from chemgraph.models.openai import _normalize_argo_model


@pytest.mark.parametrize(
    "model",
    ["argo:claude-opus-5.5", "argo:claude-sonnet-5.5", "argo:claude-haiku-5.5"],
)
@pytest.mark.parametrize(
    "base_url",
    ["http://127.0.0.1:18085/argoapi/v1", "https://proxy.example/v1"],
)
def test_claude55_custom_endpoint_strips_prefix(monkeypatch, model, base_url):
    monkeypatch.delenv("CHEMGRAPH_ARGO_MODEL_FORMAT", raising=False)

    assert _normalize_argo_model(model, base_url) == model.removeprefix("argo:")


def test_local_argo_shim_keeps_openai_style_model_name(monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_ARGO_MODEL_FORMAT", raising=False)

    assert (
        _normalize_argo_model(
            "argo:gpt-4o-mini",
            "http://127.0.0.1:18085/argoapi/v1",
        )
        == "gpt-4o-mini"
    )


def test_local_argo_shim_uses_advertised_gpt54_model_name(monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_ARGO_MODEL_FORMAT", raising=False)

    assert (
        _normalize_argo_model(
            "argo:gpt-5.4",
            "http://127.0.0.1:18085/argoapi/v1",
        )
        == "GPT-5.4"
    )


def test_hosted_argo_endpoint_uses_wire_model_name(monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_ARGO_MODEL_FORMAT", raising=False)

    assert (
        _normalize_argo_model(
            "argo:gpt-4o-mini",
            "https://apps.inside.anl.gov/argoapi/v1",
        )
        == "gpt4omini"
    )


def test_argo_model_format_env_override(monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_ARGO_MODEL_FORMAT", "openai")
    assert (
        _normalize_argo_model(
            "argo:gpt-4o-mini",
            "https://apps.inside.anl.gov/argoapi/v1",
        )
        == "gpt-4o-mini"
    )


def test_argo_model_format_shim_override_uses_local_alias(monkeypatch):
    monkeypatch.setenv("CHEMGRAPH_ARGO_MODEL_FORMAT", "shim")
    assert (
        _normalize_argo_model(
            "argo:gpt-5.4",
            "https://apps.inside.anl.gov/argoapi/v1",
        )
        == "GPT-5.4"
    )
