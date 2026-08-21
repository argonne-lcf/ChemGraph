from langchain_ollama import ChatOllama
from chemgraph.models.supported_models import supported_ollama_models


def load_ollama_model(
    model_name: str, temperature: float, base_url: str = None
) -> ChatOllama:
    """Load an Ollama chat model into LangChain.

    This function loads a local Ollama model and configures it for use with
    LangChain. It verifies that the requested model is supported before
    attempting to load it.

    Parameters
    ----------
    model_name : str
        The name of the Ollama model to load. See supported_ollama_models for list
        of supported models.
    temperature : float
        Controls the randomness of the generated text. Higher values (e.g., 0.8)
        make the output more random, while lower values (e.g., 0.2) make it more
        deterministic.
    base_url : str, optional
        Ollama server URL (e.g. from ``[api.local]`` in config.toml).
        Defaults to ChatOllama's built-in ``http://localhost:11434``.

    Returns
    -------
    ChatOllama
        An instance of LangChain's ChatOllama model.

    Raises
    ------
    ValueError
        If the specified model is not in the list of supported models.

    Notes
    -----
    The model must be installed locally using Ollama before it can be loaded.
    """
    if model_name not in supported_ollama_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {supported_ollama_models}."
        )

    kwargs = {"model": model_name, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    llm = ChatOllama(**kwargs)
    print(f"Successfully loaded model: {model_name}")
    return llm
