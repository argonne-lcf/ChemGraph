from langchain_ollama import ChatOllama

def load_ollama_model(model_name: str, temperature: float) -> ChatOllama:
    """
    Load an Ollama chat model into LangChain.

    """

    supported_models = ["llama3.2", "llama3.1"]
    if model_name not in supported_models:
        raise ValueError(f"Unsupported model '{model_name}'. Supported models are: {supported_models}.")

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
    )
    print(f"Successfully loaded model: {model_name}")
    return llm
