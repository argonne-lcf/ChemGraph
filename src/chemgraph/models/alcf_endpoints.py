import logging
import os

from langchain_openai import ChatOpenAI

from chemgraph.models.supported_models import (
    ALCF_DEFAULT_BASE_URL,
    ALCF_MINERVA_BASE_URL,
    supported_alcf_minerva_models,
    supported_alcf_models,
)

logger = logging.getLogger(__name__)


def load_alcf_model(
    model_name: str,
    base_url: str = None,
    api_key: str = None,
) -> ChatOpenAI:
    """Load a model from ALCF inference endpoints.

    ALCF endpoints use Globus OAuth for authentication.  The access token
    can be supplied directly via *api_key* or through the
    ``ALCF_ACCESS_TOKEN`` environment variable.

    See https://docs.alcf.anl.gov/services/inference-endpoints/ for setup
    instructions and https://github.com/argonne-lcf/inference-endpoints
    for the authentication helper script.

    Parameters
    ----------
    model_name : str
        The name of the model to load.  Must be in ``supported_alcf_models``.
    base_url : str, optional
        The base URL of the API endpoint.  Falls back to the base URL of the
        cluster that serves *model_name* if not provided.
    api_key : str, optional
        Globus access token.  If not provided, the function checks the
        ``ALCF_ACCESS_TOKEN`` environment variable.

    Returns
    -------
    ChatOpenAI
        An instance of LangChain's ChatOpenAI configured for the ALCF
        endpoint.

    Raises
    ------
    ValueError
        If neither *api_key* nor ``ALCF_ACCESS_TOKEN`` is available, or if
        the model is not in the supported list.
    """

    # Resolve access token ---------------------------------------------------
    if api_key is None:
        api_key = os.getenv("ALCF_ACCESS_TOKEN")

    if not api_key:
        raise ValueError(
            "ALCF access token not found. To authenticate with ALCF:\n"
            "  1. pip install globus_sdk\n"
            "  2. wget https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/"
            "refs/heads/main/inference_auth_token.py\n"
            "  3. python inference_auth_token.py authenticate\n"
            "  4. export ALCF_ACCESS_TOKEN=$(python inference_auth_token.py get_access_token)\n"
            "\n"
            "See: https://docs.alcf.anl.gov/services/inference-endpoints/#api-access"
        )

    # Validate model name ----------------------------------------------------
    if model_name not in supported_alcf_models:
        raise ValueError(
            f"Model '{model_name}' is not supported on ALCF. "
            f"Supported models: {supported_alcf_models}"
        )

    # Resolve base URL -------------------------------------------------------
    # Each ALCF cluster is a separate endpoint, so pick the one serving it.
    if not base_url:
        if model_name in supported_alcf_minerva_models:
            base_url = ALCF_MINERVA_BASE_URL
        else:
            base_url = ALCF_DEFAULT_BASE_URL

    try:
        llm = ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
        )
        logger.info(f"Successfully loaded ALCF model: {model_name} from {base_url}")
    except Exception as e:
        logger.error(f"Failed to load ALCF model '{model_name}': {e}")
        raise

    return llm
