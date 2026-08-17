from __future__ import annotations

from chemgraph.models.alcf_endpoints import load_alcf_model
from chemgraph.models.supported_models import (
    ALCF_DEFAULT_BASE_URL,
    ALCF_MINERVA_BASE_URL,
    supported_alcf_minerva_models,
    supported_alcf_models,
)
from chemgraph.utils.config_utils import (
    get_base_url_for_model_from_flat_config,
    get_base_url_for_model_from_nested_config,
)

MINERVA_MODEL = "nemotron-3-ultra"
SOPHIA_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


def test_minerva_models_are_a_subset_of_supported_alcf_models():
    assert set(supported_alcf_minerva_models) <= set(supported_alcf_models)


def test_minerva_models_use_the_minerva_cluster_url():
    assert MINERVA_MODEL in supported_alcf_minerva_models
    assert ALCF_MINERVA_BASE_URL.endswith("/resource_server/minerva/api/v1")


def test_nested_config_keeps_minerva_url_despite_sophia_setting():
    config = {"api": {"alcf": {"base_url": ALCF_DEFAULT_BASE_URL}}}

    assert (
        get_base_url_for_model_from_nested_config(MINERVA_MODEL, config)
        == ALCF_MINERVA_BASE_URL
    )


def test_flat_config_keeps_minerva_url_despite_sophia_setting():
    config = {"api_alcf_base_url": ALCF_DEFAULT_BASE_URL}

    assert (
        get_base_url_for_model_from_flat_config(MINERVA_MODEL, config)
        == ALCF_MINERVA_BASE_URL
    )


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


def test_load_alcf_model_defaults_to_the_serving_cluster():
    minerva = load_alcf_model(MINERVA_MODEL, api_key="dummy")
    sophia = load_alcf_model(SOPHIA_MODEL, api_key="dummy")

    assert minerva.openai_api_base == ALCF_MINERVA_BASE_URL
    assert sophia.openai_api_base == ALCF_DEFAULT_BASE_URL


def test_explicit_base_url_overrides_the_cluster_default():
    custom = "http://127.0.0.1:8000/v1"
    llm = load_alcf_model(MINERVA_MODEL, base_url=custom, api_key="dummy")

    assert llm.openai_api_base == custom
