"""Load OpenAI models using LangChain."""
import os
from langchain_openai import ChatOpenAI


def load_openai_model(model_name: str, temperature: float, api_key: str = None, prompt: str = None) -> ChatOpenAI:
    """
    Load an OpenAI chat model into LangChain.

    Parameters
    ----------
    model_name : str
        The name of the OpenAI chat model to load. Supported models include 
        'gpt-3.5-turbo', 'gpt-4', 'gpt-4o', and 'gpt-4o-mini'.
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
            raise ValueError("API key not provided and could not be retrieved from the environment.")

    supported_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"]
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model '{model_name}'. Supported models are: {supported_models}.")

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )
    print(f"Successfully loaded model: {model_name}")
    return llm
