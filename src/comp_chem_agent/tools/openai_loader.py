"""Load OpenAI models using LangChain."""

import logging
import os
from getpass import getpass
from langchain_openai import ChatOpenAI
from comp_chem_agent.models.supported_models import supported_openai_models
from comp_chem_agent.utils.logging_config import setup_logger

logger = setup_logger(__name__)


def load_openai_model(
    model_name: str, temperature: float, api_key: str = None, prompt: str = None
) -> ChatOpenAI:
    """
    Load an OpenAI chat model into LangChain.

    Parameters
    ----------
    model_name : str
        The name of the OpenAI chat model to load. See supported_openai_models for list
        of supported models.
    temperature : float
        Controls the randomness of the generated text. A higher temperature results
        in more random outputs, while a lower temperature results in more deterministic outputs.
    api_key : str, optional
        The OpenAI API key. If not provided, the function will attempt to retrieve it
        from the environment variable `OPENAI_API_KEY`.

    Returns
    -------
    ChatOpenAI
        An instance of LangChain's ChatOpenAI model.

    Raises
    ------
    ValueError
        If the API key is not provided and cannot be retrieved from the environment.

    Notes
    -----
    Ensure the model_name provided is one of the supported models. Unsupported models
    will result in an exception.
    """

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.info("OpenAI API key not found in environment variables.")
            api_key = getpass("Please enter your OpenAI API key: ")
            os.environ["OPENAI_API_KEY"] = api_key

    if model_name not in supported_openai_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {supported_openai_models}."
        )

    try:
        logger.info(f"Loading OpenAI model: {model_name}")
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
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
            return load_openai_model(model_name, temperature, api_key, prompt)
        else:
            logger.error(f"Error loading OpenAI model: {str(e)}")
            raise
