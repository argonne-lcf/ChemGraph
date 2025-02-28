from langchain_ollama import ChatOllama
from comp_chem_agent.models.supported_models import supported_ollama_models


def load_ollama_model(model_name: str, temperature: float) -> ChatOllama:
    """
    Load an Ollama chat model into LangChain.

    Parameters
    ----------
    model_name : str
        The name of the Ollama model to load. See supported_ollama_models for list
        of supported models.
    temperature : float
        Controls the randomness of the generated text.

    Returns
    -------
    ChatOllama
        An instance of LangChain's ChatOllama model.
    """

    if model_name not in supported_ollama_models:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported models are: {supported_ollama_models}."
        )

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )
    print(f"Successfully loaded model: {model_name}")
    return llm
