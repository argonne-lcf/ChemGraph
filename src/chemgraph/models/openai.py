"""Load OpenAI models using LangChain."""

import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from chemgraph.models.supported_models import (
    ARGO_DEFAULT_BASE_URL,
    supported_openai_models,
    supported_argo_models,
)
from chemgraph.utils.config_utils import normalize_openai_base_url
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Maps user-facing ``argo:`` model names to the internal wire names
# expected by the Argo API (https://apps.inside.anl.gov/argoapi).
# When a different endpoint (e.g. ArgoProxy) is used, the ``argo:``
# prefix is stripped instead and the remainder is sent as-is.
ARGO_MODEL_MAP = {
    # GPT family
    "argo:gpt-3.5-turbo": "gpt35",
    "argo:gpt-3.5-turbo-16k": "gpt35turbo16k",
    "argo:gpt-4": "gpt4",
    "argo:gpt-4-32k": "gpt432k",
    "argo:gpt-4-turbo": "gpt4turbo",
    "argo:gpt-4o": "gpt4o",
    "argo:gpt-4o-latest": "gpt4olatest",
    "argo:gpt-4o-mini": "gpt4omini",
    "argo:gpt-4.1": "gpt41",
    "argo:gpt-4.1-mini": "gpt41mini",
    "argo:gpt-4.1-nano": "gpt41nano",
    "argo:gpt-5": "gpt5",
    "argo:gpt-5-mini": "gpt5mini",
    "argo:gpt-5-nano": "gpt5nano",
    "argo:gpt-5.1": "gpt51",
    "argo:gpt-5.2": "gpt52",
    "argo:gpt-5.4": "gpt54",

    # Reasoning / o-series
    "argo:o1-preview": "gpto1preview",
    "argo:o1-mini": "gpto1mini",
    "argo:o1": "gpto1",
    "argo:o3-mini": "gpto3mini",
    "argo:o3": "gpto3",
    "argo:o4-mini": "gpto4mini",
    # Gemini via Argo
    "argo:gemini-2.5-pro": "gemini25pro",
    "argo:gemini-2.5-flash": "gemini25flash",
    # Claude via Argo
    "argo:claude-opus-4.6": "claudeopus46",
    "argo:claude-opus-4.5": "claudeopus45",
    "argo:claude-opus-4.1": "claudeopus41",
    "argo:claude-opus-4": "claudeopus4",
    "argo:claude-haiku-4.5": "claudehaiku45",
    "argo:claude-sonnet-4.5": "claudesonnet45",
    "argo:claude-sonnet-4": "claudesonnet4",
    "argo:claude-sonnet-3.5-v2": "claudesonnet35v2",
    "argo:claude-haiku-3.5": "claudehaiku35",
}


def _normalize_argo_model(model_name: str, base_url: str) -> str:
    """Normalize an ``argo:``-prefixed model name for the target endpoint.

    * Argo API (base_url contains ``argoapi``): map to internal wire
      names via ``ARGO_MODEL_MAP`` (e.g. ``argo:gpt-4o`` -> ``gpt4o``).
    * Other endpoints (ArgoProxy, custom): strip the ``argo:`` prefix
      and send the remainder as-is (e.g. ``argo:gpt-4o`` -> ``gpt-4o``).
    """
    if not model_name.startswith("argo:"):
        return model_name

    if base_url and "argoapi" in base_url:
        # Argo API endpoint -- use the wire-name map
        normalized = ARGO_MODEL_MAP.get(model_name)
        if normalized:
            logger.info("Normalized Argo model '%s' -> '%s'", model_name, normalized)
            return normalized
        # Fallback: strip prefix and remove punctuation
        fallback = model_name.removeprefix("argo:").replace("-", "").replace(".", "")
        logger.info(
            "Normalized Argo model '%s' -> '%s' (fallback)", model_name, fallback
        )
        return fallback
    else:
        # Non-Argo-API endpoint -- strip prefix only
        stripped = model_name.removeprefix("argo:")
        logger.info("Stripped argo: prefix '%s' -> '%s'", model_name, stripped)
        return stripped


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

    # Apply default Argo base URL for argo: models when none is specified.
    if model_name.startswith("argo:") and not base_url:
        base_url = ARGO_DEFAULT_BASE_URL
        logger.info("Using default Argo base URL: %s", base_url)

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if model_name.startswith("argo:"):
                # Argo API authenticates via the 'user' field, not an API key.
                # Use argo_user as a placeholder since ChatOpenAI requires a value.
                api_key = argo_user or os.getenv("ARGO_USER", "chemgraph")
            else:
                logger.warning("OPENAI_API_KEY not found in environment variables.")
                print("OPENAI_API_KEY not set. Please enter your OpenAI API key.")
                api_key = getpass("OpenAI API key: ")
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
                llm_kwargs["model_kwargs"] = {"user": argo_user}
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
            print("The provided OpenAI API key is invalid. Please enter a valid key.")
            api_key = getpass("OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
            # Retry with new API key
            return load_openai_model(
                model_name, temperature, api_key, prompt, base_url, argo_user
            )
        else:
            logger.error(f"Error loading OpenAI model: {str(e)}")
            raise
