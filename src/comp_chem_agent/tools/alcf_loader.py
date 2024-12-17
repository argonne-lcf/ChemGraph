import os
from langchain_openai import ChatOpenAI


def load_alcf_model(model_name: str, base_url: str, api_key: str = None) -> ChatOpenAI:
    """
    Load an models from ALCF inference endpoints (https://github.com/argonne-lcf/inference-endpoints).

    Parameters
    ----------
    model_name : str
        The name of the OpenAI chat model to load.
    base_url : str
        The base URL of the API endpoint.
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
    """

    if api_key is None:
        raise ValueError("API key (access token) is not found")
    supported_models = [
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.3-70B-Instruct"
    ]
    
    if model_name not in supported_models:
        raise ValueError(f"Model {model_name} is not supported on ALCF yet.")
    try:
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        print(f"Successfully loaded model: {model_name} from {base_url}")

    except Exception as e:
        print(f"Error with loading {model_name}")
        print(e)

    return llm