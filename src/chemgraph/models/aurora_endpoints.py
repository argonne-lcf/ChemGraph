import logging
import os

from langchain_openai import ChatOpenAI

from chemgraph.models.supported_models import AURORA_DEFAULT_BASE_URL

logger = logging.getLogger(__name__)

AURORA_MODEL_PREFIX = "aurora:"


def _normalize_aurora_model(model_name: str) -> str:
    """Strip the ``aurora:`` prefix to get the name the endpoint expects.

    Aurora on-node servers advertise each model under its own id
    (llama-server ``--alias`` / vLLM ``--served-model-name``). The prefix only
    selects the provider inside ChemGraph, so it is removed before the request
    is sent.

    Parameters
    ----------
    model_name : str
        Requested model identifier, normally ``aurora:``-prefixed.

    Returns
    -------
    str
        Model name to send to the endpoint.
    """
    if not model_name.startswith(AURORA_MODEL_PREFIX):
        return model_name

    stripped = model_name.removeprefix(AURORA_MODEL_PREFIX)
    logger.info("Stripped aurora: prefix '%s' -> '%s'", model_name, stripped)
    return stripped


def load_aurora_model(
    model_name: str,
    base_url: str = None,
    api_key: str = None,
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Load a model from an Aurora on-node inference server.

    Aurora LLM servers (llama.cpp SYCL ``llama-server`` or vLLM-XPU) expose an
    OpenAI-compatible ``/v1`` endpoint. Because a compute node has no public IP
    and its address changes per job, the base URL is usually supplied via
    *base_url*, the ``AURORA_BASE_URL`` environment variable, or
    ``[api.aurora].base_url`` in the config; it falls back to
    ``AURORA_DEFAULT_BASE_URL`` (a co-located ``127.0.0.1`` server).

    These servers typically do not enforce authentication, so *api_key* defaults
    to ``"dummy"`` (``langchain_openai`` requires a non-empty value).

    The selected model MUST support OpenAI tool calling; ChemGraph's workflows
    are tool-driven.

    Parameters
    ----------
    model_name : str
        Model identifier, normally ``aurora:``-prefixed. The remainder must
        match the server's advertised model id.
    base_url : str, optional
        Endpoint base URL. Falls back to ``AURORA_BASE_URL`` then
        ``AURORA_DEFAULT_BASE_URL``.
    api_key : str, optional
        API key. Falls back to ``AURORA_API_KEY`` / ``OPENAI_API_KEY`` then
        ``"dummy"``.
    temperature : float, optional
        Sampling temperature, by default 0.0.

    Returns
    -------
    ChatOpenAI
        A LangChain ``ChatOpenAI`` configured for the Aurora endpoint.
    """
    if not base_url:
        base_url = os.getenv("AURORA_BASE_URL") or AURORA_DEFAULT_BASE_URL

    if not api_key:
        api_key = (
            os.getenv("AURORA_API_KEY") or os.getenv("OPENAI_API_KEY") or "dummy"
        )

    wire_model = _normalize_aurora_model(model_name)

    try:
        llm = ChatOpenAI(
            model=wire_model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
        logger.info(
            "Successfully loaded Aurora model: %s from %s", wire_model, base_url
        )
    except Exception as e:
        logger.error("Failed to load Aurora model '%s': %s", model_name, e)
        raise

    return llm
