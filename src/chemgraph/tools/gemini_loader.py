"""Load Gemini models using LangChain."""

import os
from getpass import getpass
from langchain_google_genai import ChatGoogleGenerativeAI
from chemgraph.models.supported_models import supported_gemini_models
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_gemini_model(
    model_name: str,
    temperature: float,
    api_key: str = None,
    prompt: str = None,
    base_url: str = None,
) -> ChatGoogleGenerativeAI:
    """Load an Gemini chat model into LangChain.

    This function loads an Gemini model and configures it for use with LangChain.
    It handles API key management, including prompting for the key if not provided
    or if the provided key is invalid.

    Parameters
    ----------
    model_name : str
        The name of the Gemini chat model to load. See supported_gemini_models for list
        of supported models.
    temperature : float
        Controls the randomness of the generated text. Higher values (e.g., 0.8)
        make the output more random, while lower values (e.g., 0.2) make it more
        deterministic.
    api_key : str, optional
        The Google API key. If not provided, the function will attempt to retrieve it
        from the environment variable `GOOGLE_API_KEY`.
    prompt : str, optional
        Custom prompt to use when requesting the API key from the user.

    Returns
    -------
    ChatGoogleGenerativeAI
        An instance of LangChain's ChatGoogleGenerativeAI model.

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
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.info("Google API key not found in environment variables.")
            api_key = getpass("Please enter your Google API key: ")
            os.environ["GOOGLE_API_KEY"] = api_key

    if model_name not in supported_gemini_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {supported_gemini_models}."
        )

    try:
        logger.info(f"Loading Gemini model: {model_name}")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            max_output_tokens=5000,
        )
        # No guarantee that api_key is valid, authentication happens only during invocation
        logger.info(f"Requested model: {model_name}")
        logger.info("Gemini model loaded successfully")
        return llm
    except Exception as e:
        # Can remove this since authentication happens only during invocation
        if "AuthenticationError" in str(e) or "invalid_api_key" in str(e):
            logger.warning("Invalid Google API key.")
            api_key = getpass("Please enter a valid Google API key: ")
            os.environ["GOOGLE_API_KEY"] = api_key
            # Retry with new API key
            return load_gemini_model(model_name, temperature, api_key, prompt)
        else:
            logger.error(f"Error loading Google model: {str(e)}")
            raise
