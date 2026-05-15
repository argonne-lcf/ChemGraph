"""Load Gemini models using LangChain."""

import os
from getpass import getpass
from langchain_surf import ChatWillma
from chemgraph.models.supported_models import supported_surfai_models
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_willma_model(
    model_name: str,
    temperature: float,
    api_key: str = None,
) -> ChatWillma:
    """Load an Willma chat model into LangChain.

    This function loads an Willma model and configures it for use with LangChain.
    It handles API key management, including prompting for the key if not provided
    or if the provided key is invalid.

    Parameters
    ----------
    model_name : str
        The name of the Willma chat model to load. See supported_willma_models for list
        of supported models.
    temperature : float
        Controls the randomness of the generated text. Higher values (e.g., 0.8)
        make the output more random, while lower values (e.g., 0.2) make it more
        deterministic.
    api_key : str, optional
        The Willma API key. If not provided, the function will attempt to retrieve it
        from the environment variable `WILLMA_API_KEY`.
    prompt : str, optional
        Custom prompt to use when requesting the API key from the user.

    Returns
    -------
    ChatWillma
        An instance of LangChain's ChatWillma model.

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

    if api_key is None:
        api_key = os.getenv("AIHUB_API_KEY")
        if not api_key:
            logger.info("Willma API key not found in environment variables.")
            api_key = getpass("Please enter your Willma API key: ")
            os.environ["AIHUB_API_KEY"] = api_key

    if model_name not in supported_surfai_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {supported_surfai_models}."
        )

    try:
        logger.info(f"Loading Willma model: {model_name}")
        llm = ChatWillma(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            # max_output_tokens=6000,
        )
        # No guarantee that api_key is valid, authentication happens only during invocation
        logger.info(f"Requested model: {model_name}")
        logger.info("Willma model loaded successfully")
        return llm
    except Exception as e:
        # Can remove this since authentication happens only during invocation
        if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
            logger.warning("Invalid Willma API key.")
            api_key = getpass("Please enter a valid Willma API key: ")
            os.environ["AIHUB_API_KEY"] = api_key
            # Retry with new API key
            return load_willma_model(model_name, temperature, api_key)
        else:
            logger.error(f"Error loading Willma model: {str(e)}")
            raise
