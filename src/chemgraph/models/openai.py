"""Load OpenAI models using LangChain."""

import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from chemgraph.models.supported_models import (
    supported_openai_models,
    supported_argo_models,
)
from chemgraph.utils.config_utils import normalize_openai_base_url
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

ARGO_MODEL_MAP = {
    "argo:gpt-4o": "gpt4o",
    "argo:gpt-4o-latest": "gpt4olatest",
    "argo:gpt-4.1": "gpt41",
    "argo:gpt-4.1-mini": "gpt41mini",
    "argo:gpt-4.1-nano": "gpt41nano",
    "argo:gpt-o1-preview": "gpto1preview",
    "argo:o1-preview": "gpto1preview",
    "argo:gpt-o1-mini": "gpto1mini",
    "argo:o1-mini": "gpto1mini",
    "argo:gpt-o3-mini": "gpto3mini",
    "argo:o3-mini": "gpto3mini",
    "argo:gpt-o1": "gpto1",
    "argo:o1": "gpto1",
    "argo:gpt-o3": "gpto3",
    "argo:o3": "gpto3",
    "argo:gpt-o4-mini": "gpto4mini",
    "argo:o4-mini": "gpto4mini",
}


def _normalize_argo_model(model_name: str, base_url: str) -> str:
    if not base_url or "argoapi" not in base_url:
        return model_name
    normalized = ARGO_MODEL_MAP.get(model_name, model_name)
    if normalized != model_name:
        logger.info("Normalized Argo model '%s' -> '%s'", model_name, normalized)
    return normalized


def load_openai_model(
    model_name: str,
    temperature: float,
    api_key: str = None,
    prompt: str = None,
    base_url: str = None,
    argo_user: str = None,
) -> ChatOpenAI:
    """Load an OpenAI chat model into LangChain.

    This function loads an OpenAI model and configures it for use with LangChain.
    It handles API key management, including prompting for the key if not provided
    or if the provided key is invalid.

    Parameters
    ----------
    model_name : str
        The name of the OpenAI chat model to load. See supported_openai_models for list
        of supported models.
    temperature : float
        Controls the randomness of the generated text. Higher values (e.g., 0.8)
        make the output more random, while lower values (e.g., 0.2) make it more
        deterministic.
    api_key : str, optional
        The OpenAI API key. If not provided, the function will attempt to retrieve it
        from the environment variable `OPENAI_API_KEY`.
    prompt : str, optional
        Custom prompt to use when requesting the API key from the user.

    Returns
    -------
    ChatOpenAI
        An instance of LangChain's ChatOpenAI model.

    Raises
    ------
    ValueError
        If the model name is not in the list of supported models.
    Exception
        If there is an error loading the model or if the API key is invalid.

    Notes
    -----
    The function will:
    1. Check for the API key in the environment variables
    2. Prompt for the key if not found
    3. Validate the model name against supported models
    4. Attempt to load the model
    5. Handle any authentication errors by prompting for a new key
    """

    base_url = normalize_openai_base_url(base_url)

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.info("OpenAI API key not found in environment variables.")
            api_key = getpass("Please enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key

    if model_name not in supported_openai_models and model_name not in supported_argo_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {supported_openai_models}."
        )

    is_argo_endpoint = bool(base_url and "argoapi" in base_url)
    argo_user = (
        argo_user or os.getenv("ARGO_USER", "chemgraph")
        if is_argo_endpoint
        else None
    )

    try:
        if base_url is not None:
            logger.info(f"Using custom base URL: {base_url}")
            model_name = _normalize_argo_model(model_name, base_url)
            llm_kwargs = dict(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                max_tokens=4000,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            # Argo gateways may require an explicit "user" field in payload.
            if is_argo_endpoint and argo_user:
                llm_kwargs["user"] = argo_user
                logger.info(
                    "Using Argo user from config/ARGO_USER/default: %s", argo_user
                )
            llm = ChatOpenAI(**llm_kwargs)
        else:
            logger.info(f"Loading OpenAI model: {model_name}")
            llm = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                max_tokens=6000,
            )
        # No guarantee that api_key is valid, authentication happens only during invocation
        logger.info(f"Requested model: {model_name}")
        logger.info("OpenAI model loaded successfully")
        return llm
    except Exception as e:
        # Can remove this since authentication happens only during invocation
        if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
            logger.warning("Invalid OpenAI API key.")
            api_key = getpass("Please enter a valid OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
            # Retry with new API key
            return load_openai_model(
                model_name, temperature, api_key, prompt, base_url, argo_user
            )
        else:
            logger.error(f"Error loading OpenAI model: {str(e)}")
            raise
