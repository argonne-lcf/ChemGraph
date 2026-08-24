"""Ordered endpoint registry and side-effect-free model selection helpers."""

from __future__ import annotations

from chemgraph.models.endpoints import EndpointSpec, ModelRequest
from chemgraph.models.endpoints import alcf as alcf_ep
from chemgraph.models.endpoints import anthropic_direct as anthropic_ep
from chemgraph.models.endpoints import argo as argo_ep
from chemgraph.models.endpoints import codex as codex_ep
from chemgraph.models.endpoints import google_direct as google_ep
from chemgraph.models.endpoints import groq as groq_ep
from chemgraph.models.endpoints import ollama as ollama_ep
from chemgraph.models.endpoints import openai_direct as openai_ep
from chemgraph.models.endpoints import openrouter as openrouter_ep
from chemgraph.models.endpoints import vllm as vllm_ep

# Prefix routes precede catalog matches. vLLM remains a configured last resort.
ENDPOINT_REGISTRY = (
    codex_ep.SPEC,
    openrouter_ep.SPEC,
    argo_ep.ANTHROPIC_SPEC,
    argo_ep.OPENAI_SPEC,
    alcf_ep.SPEC,
    openai_ep.SPEC,
    anthropic_ep.SPEC,
    google_ep.SPEC,
    ollama_ep.SPEC,
    groq_ep.SPEC,
)

# Preserve the historical discovery order independently of dispatch order.
CATALOG_ENDPOINTS = (
    openai_ep.SPEC,
    ollama_ep.SPEC,
    alcf_ep.SPEC,
    anthropic_ep.SPEC,
    argo_ep.OPENAI_SPEC,
    google_ep.SPEC,
    groq_ep.SPEC,
    openrouter_ep.SPEC,
)


def match_endpoint(model: str) -> EndpointSpec | None:
    """Return the first model-matched endpoint, excluding the vLLM fallback."""
    for spec in ENDPOINT_REGISTRY:
        if spec.matches(model):
            return spec
    return None


def select_endpoint(request: ModelRequest) -> EndpointSpec:
    """Select a model endpoint without preparing credentials or a client."""
    spec = match_endpoint(request.model)
    if spec is not None:
        return spec

    if vllm_ep.can_handle(request):
        return vllm_ep.SPEC

    raise ValueError(
        f"Model '{request.model}' not found in any supported model list. "
        "Use a model from: OpenAI, Anthropic, Gemini, groq:<model>, "
        "openrouter:<model>, codex:<model>, argo:<model>, ALCF, or Ollama. "
        "For a custom OpenAI-compatible model, provide a base URL or configure "
        "[api.vllm].base_url."
    )


def catalog_entries() -> list[tuple[str, EndpointSpec]]:
    """Return deduplicated curated model/spec pairs in display order."""
    entries: list[tuple[str, EndpointSpec]] = []
    seen: set[str] = set()
    for spec in CATALOG_ENDPOINTS:
        for model in spec.curated_models:
            if model not in seen:
                seen.add(model)
                entries.append((model, spec))
    return entries


def catalog_models() -> list[str]:
    """Return the curated model catalog in compatibility display order."""
    return [model for model, _spec in catalog_entries()]


def config_sections() -> tuple[str, ...]:
    """Return all canonical and legacy endpoint configuration section names."""
    sections: list[str] = []
    for spec in (*ENDPOINT_REGISTRY, vllm_ep.SPEC):
        for name in (spec.config_section, *spec.legacy_config_sections):
            if name and name not in sections:
                sections.append(name)
    return tuple(sections)
