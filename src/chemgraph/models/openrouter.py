"""Load OpenRouter models using LangChain's OpenAI-compatible client."""

import os
import sys
from getpass import getpass

from langchain_openai import ChatOpenAI

from chemgraph.models.supported_models import (
    MODELS_WITHOUT_TEMPERATURE,
    OPENROUTER_DEFAULT_BASE_URL,
)
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)

# Higher than the OpenAI/Groq defaults on purpose: most models served through
# OpenRouter emit reasoning tokens by default, and those count against the
# *completion* budget. Too small a cap lets chain-of-thought exhaust it before
# the tool call is emitted, producing an empty turn the graph reads as a dead end.
OPENROUTER_DEFAULT_MAX_TOKENS = 8000


def load_openrouter_model(
    model_name: str,
    temperature: float = 0.0,
    api_key: str = None,
    base_url: str = None,
    max_tokens: int = OPENROUTER_DEFAULT_MAX_TOKENS,
) -> ChatOpenAI:
    """Load an OpenRouter chat model into LangChain.

    OpenRouter exposes an OpenAI-compatible API, so this returns a
    ``ChatOpenAI`` pointed at the OpenRouter endpoint. Any slug from
    https://openrouter.ai/models works; ``supported_openrouter_models`` is a
    discovery list, not a gate.

    Parameters
    ----------
    model_name : str
        Model identifier with the ``openrouter:`` prefix. The prefix is
        stripped and the remainder sent verbatim (e.g. ``moonshotai/kimi-k3``).
    temperature : float
        Sampling temperature. Omitted for ``MODELS_WITHOUT_TEMPERATURE`` entries.
    api_key : str, optional
        Falls back to ``OPENROUTER_API_KEY``. Never to ``OPENAI_API_KEY``.
    base_url : str, optional
        Endpoint override. Defaults to ``OPENROUTER_DEFAULT_BASE_URL``.
    max_tokens : int
        Completion-token budget. See ``OPENROUTER_DEFAULT_MAX_TOKENS``.

    Returns
    -------
    ChatOpenAI
        A LangChain chat model bound to the OpenRouter endpoint.

    Raises
    ------
    ValueError
        If no API key is available and there is no terminal to prompt on.
    """
    # Keep the prefixed name -- it is what the quirk sets are keyed on.
    requested_model_name = model_name
    model_name = model_name.removeprefix("openrouter:")

    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        # Only prompt when there is a terminal. Under the eval harness
        # (nohup/cron) getpass() would not fail -- it would block on stdin
        # forever, with no traceback and no exit code.
        if sys.stdin.isatty():
            logger.info("OpenRouter API key not found in environment variables.")
            api_key = getpass("Please enter your OpenRouter API key: ")
            os.environ["OPENROUTER_API_KEY"] = api_key
        else:
            raise ValueError(
                "OpenRouter API key not found. Set the OPENROUTER_API_KEY "
                "environment variable:\n"
                "  export OPENROUTER_API_KEY='your_key_here'\n"
                "  Get a key at: https://openrouter.ai/keys"
            )

    logger.info("Loading OpenRouter model: %s", model_name)
    llm_kwargs = dict(
        model=model_name,
        api_key=api_key,
        base_url=base_url or OPENROUTER_DEFAULT_BASE_URL,
        max_tokens=max_tokens,
    )
    # top_p / frequency_penalty / presence_penalty are deliberately not sent:
    # they are no-ops at temperature 0, and OpenRouter fans requests out to a
    # rotating pool of upstream providers whose real parameter support is
    # narrower than the advertised union.
    if requested_model_name not in MODELS_WITHOUT_TEMPERATURE:
        llm_kwargs["temperature"] = temperature

    # Authentication happens only during invocation.
    return ChatOpenAI(**llm_kwargs)
