from __future__ import annotations

import pytest

from chemgraph.models.alcf_endpoints import _normalize_alcf_model, load_alcf_model
from chemgraph.models.supported_models import (
    ALCF_DEFAULT_BASE_URL,
    ALCF_METIS_BASE_URL,
    ALCF_MINERVA_BASE_URL,
    supported_alcf_metis_models,
    supported_alcf_minerva_models,
    supported_alcf_models,
)
from chemgraph.utils.config_utils import (
    get_base_url_for_model_from_flat_config,
    get_base_url_for_model_from_nested_config,
)

SOPHIA_MODEL = "alcf:meta-llama/Llama-3.3-70B-Instruct"
MINERVA_MODEL = "alcf:nemotron-3-ultra"
METIS_MODEL = "alcf:Mistral-Large-3-675B-Instruct-2512"


def test_every_alcf_model_carries_the_prefix():
    assert all(m.startswith("alcf:") for m in supported_alcf_models)


def test_cluster_lists_are_subsets_of_supported_alcf_models():
    known = set(supported_alcf_models)
    assert set(supported_alcf_minerva_models) <= known
    assert set(supported_alcf_metis_models) <= known


def test_cluster_lists_do_not_overlap():
    assert not set(supported_alcf_minerva_models) & set(supported_alcf_metis_models)


def test_supported_alcf_models_has_no_duplicates():
    assert len(supported_alcf_models) == len(set(supported_alcf_models))


def test_normalize_strips_the_prefix():
    assert _normalize_alcf_model(MINERVA_MODEL) == "nemotron-3-ultra"
    assert (
        _normalize_alcf_model(SOPHIA_MODEL) == "meta-llama/Llama-3.3-70B-Instruct"
    )


def test_normalize_leaves_unprefixed_names_alone():
    assert _normalize_alcf_model("nemotron-3-ultra") == "nemotron-3-ultra"


def test_metis_and_sophia_keep_similar_names_apart():
    # Metis serves "gpt-oss-120b"; Sophia serves "openai/gpt-oss-120b".
    assert "alcf:gpt-oss-120b" in supported_alcf_metis_models
    assert "alcf:openai/gpt-oss-120b" not in supported_alcf_metis_models

    assert (
        get_base_url_for_model_from_nested_config("alcf:gpt-oss-120b", {})
        == ALCF_METIS_BASE_URL
    )
    assert (
        get_base_url_for_model_from_nested_config("alcf:openai/gpt-oss-120b", {})
        == ALCF_DEFAULT_BASE_URL
    )


def test_non_sophia_models_keep_their_url_despite_a_sophia_setting():
    nested = {"api": {"alcf": {"base_url": ALCF_DEFAULT_BASE_URL}}}
    flat = {"api_alcf_base_url": ALCF_DEFAULT_BASE_URL}

    for model, expected in (
        (MINERVA_MODEL, ALCF_MINERVA_BASE_URL),
        (METIS_MODEL, ALCF_METIS_BASE_URL),
    ):
        assert get_base_url_for_model_from_nested_config(model, nested) == expected
        assert get_base_url_for_model_from_flat_config(model, flat) == expected


def test_sophia_models_still_honour_the_configured_alcf_url():
    custom = "https://example.invalid/resource_server/sophia/vllm/v1"

    assert (
        get_base_url_for_model_from_nested_config(
            SOPHIA_MODEL, {"api": {"alcf": {"base_url": custom}}}
        )
        == custom
    )
    assert (
        get_base_url_for_model_from_flat_config(
            SOPHIA_MODEL, {"api_alcf_base_url": custom}
        )
        == custom
    )


def test_sophia_models_fall_back_to_the_default_url():
    assert (
        get_base_url_for_model_from_nested_config(SOPHIA_MODEL, {})
        == ALCF_DEFAULT_BASE_URL
    )
    assert (
        get_base_url_for_model_from_flat_config(SOPHIA_MODEL, {})
        == ALCF_DEFAULT_BASE_URL
    )


def test_load_alcf_model_uses_the_serving_cluster_and_strips_the_prefix():
    for model, expected_url, expected_wire in (
        (SOPHIA_MODEL, ALCF_DEFAULT_BASE_URL, "meta-llama/Llama-3.3-70B-Instruct"),
        (MINERVA_MODEL, ALCF_MINERVA_BASE_URL, "nemotron-3-ultra"),
        (METIS_MODEL, ALCF_METIS_BASE_URL, "Mistral-Large-3-675B-Instruct-2512"),
    ):
        llm = load_alcf_model(model, api_key="dummy")
        assert llm.openai_api_base == expected_url
        assert llm.model_name == expected_wire


def test_explicit_base_url_overrides_the_cluster_default():
    custom = "http://127.0.0.1:8000/v1"
    llm = load_alcf_model(MINERVA_MODEL, base_url=custom, api_key="dummy")

    assert llm.openai_api_base == custom
    assert llm.model_name == "nemotron-3-ultra"


def test_unprefixed_model_names_are_rejected():
    with pytest.raises(ValueError, match="not supported on ALCF"):
        load_alcf_model("meta-llama/Llama-3.3-70B-Instruct", api_key="dummy")
