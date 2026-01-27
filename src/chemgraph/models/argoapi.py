"""Load OpenAI models using LangChain."""

import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from chemgraph.models.supported_models import supported_argo_models
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_argoapi_model(
    model_name: str,
    temperature: float,
    api_key: str = None,
    prompt: str = None,
    base_url: str = None,
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

    if model_name not in supported_argo_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {supported_argo_models}."
        )

    try:
        if base_url is not None:
            logger.info(f"Using custom base URL: {base_url}")
            llm = ChatOpenAI(
                user="tpham",
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                max_tokens=4000,
            )
            return llm

        else:
            raise ValueError(
                f"Unsupported model '{model_name}'. Supported models are: {supported_argo_models}."
            )
    except Exception as e:
        # Can remove this since authentication happens only during invocation
        if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
            logger.warning("Invalid OpenAI API key.")
            api_key = getpass("Please enter a valid OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key
            # Retry with new API key
            return load_argoapi_model(model_name, temperature, api_key, prompt)
        else:
            logger.error(f"Error loading OpenAI model: {str(e)}")
            raise
