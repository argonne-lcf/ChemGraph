"""Shared model-loading utility for ChemGraph.

Provides a single ``load_chat_model`` function that detects the provider
for a given model name and returns a LangChain ``BaseChatModel`` instance.
This avoids duplicating provider-detection logic across the agent and
evaluation modules.
"""

from typing import Optional

from chemgraph.models.alcf_endpoints import load_alcf_model
from chemgraph.models.anthropic import load_anthropic_model
from chemgraph.models.gemini import load_gemini_model
from chemgraph.models.groq import load_groq_model
from chemgraph.models.local_model import load_ollama_model
from chemgraph.models.openai import load_openai_model
from chemgraph.models.supported_models import (
    supported_alcf_models,
    supported_anthropic_models,
    supported_argo_models,
    supported_gemini_models,

    supported_ollama_models,
    supported_openai_models,
)


def load_chat_model(
    model_name: str,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    argo_user: Optional[str] = None,
):
    """Load a LangChain chat model by provider auto-detection.

    Parameters
    ----------
    model_name : str
        Model name from any supported provider list.
    temperature : float
        Sampling temperature (default 0.0 for deterministic output).
    base_url : str, optional
        Provider base URL override.
    api_key : str, optional
        API key override (falls back to environment variables).
    argo_user : str, optional
        Argo user identifier.

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance.

    Raises
    ------
    ValueError
        If the model name is not found in any supported provider list.
    """
    if model_name in supported_openai_models or model_name in supported_argo_models:
        kwargs = {
            "model_name": model_name,
            "temperature": temperature,
            "base_url": base_url,
        }
        if argo_user is not None:
            kwargs["argo_user"] = argo_user
        return load_openai_model(**kwargs)
    elif model_name in supported_ollama_models:
        return load_ollama_model(model_name=model_name, temperature=temperature)
    elif model_name in supported_alcf_models:
        return load_alcf_model(
            model_name=model_name, base_url=base_url, api_key=api_key
        )
    elif model_name in supported_anthropic_models:
        return load_anthropic_model(
            model_name=model_name, api_key=api_key, temperature=temperature
        )
    elif model_name in supported_gemini_models:
        return load_gemini_model(
            model_name=model_name, api_key=api_key, temperature=temperature
        )
    elif model_name.startswith("groq:"):
        return load_groq_model(
            model_name=model_name, api_key=api_key, temperature=temperature
        )
    else:
        raise ValueError(
            f"Model '{model_name}' not found in any supported model list. "
            f"Use a model from: OpenAI, Anthropic, Gemini, groq:<model>, argo:<model>, ALCF, or Ollama."
        )
