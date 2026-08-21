"""Shared, internal types for the endpoint × protocol model-loading pipeline.

The pipeline is::

    caller -> load_chat_model -> endpoint spec -> protocol builder -> client

An *endpoint* declares only what makes it different from other endpoints that
speak the same protocol: its base URL, credential policy, model-name transform,
default parameters, and per-model quirks. A *protocol* owns the single
construction site for one LangChain client class.

Everything in this module is internal. The public model-loading surface remains
``load_chat_model``, ``LLMSettings``, and ``ChemGraph``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from getpass import getpass
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import urlparse

from chemgraph.utils.logging_config import setup_logger

if TYPE_CHECKING:  # avoid a runtime import cycle with settings.py
    from langchain_core.language_models.chat_models import BaseChatModel

    from chemgraph.models.settings import LLMSettings

logger = setup_logger(__name__)


@dataclass(frozen=True)
class ModelRequest:
    """A fully-formed request to load one chat model.

    Built by the loader from either explicit arguments or an ``LLMSettings``
    object and handed to an endpoint's ``prepare`` callback.
    """

    model: str
    temperature: float = 0.0
    base_url: str | None = None
    api_key: str | None = None
    argo_user: str | None = None
    reasoning_effort: str | None = None
    settings: "LLMSettings | None" = None


@dataclass(frozen=True)
class CredentialPolicy:
    """How an endpoint resolves its API key.

    Credentials never cross provider boundaries. The one exception -- the vLLM
    fallback to ``OPENAI_API_KEY`` -- is expressed explicitly on that endpoint,
    not encoded here.
    """

    env_var: str | None = None
    required: bool = True
    interactive_prompt: bool = False
    missing_key_help: str | None = None
    placeholder: str | None = None


@dataclass(frozen=True)
class PreparedModel:
    """The result of an endpoint preparing a request.

    Holds everything the protocol builder needs plus caller-visible metadata
    the loader can surface without re-deriving provider facts.
    """

    endpoint_name: str
    protocol: str
    client_kwargs: dict[str, Any]
    reasoning_effort: str | None = None
    supports_structured_output: bool = True


@dataclass(frozen=True)
class EndpointSpec:
    """Declarative description of one endpoint route.

    ``matches`` decides whether this spec handles a model name; ``prepare``
    turns a ``ModelRequest`` into a ``PreparedModel``; ``protocol_build`` builds
    the LangChain client from ``PreparedModel.client_kwargs``.
    """

    name: str
    protocol: str
    matches: Callable[[str], bool]
    prepare: Callable[[ModelRequest], PreparedModel]
    protocol_build: Callable[[dict[str, Any]], "BaseChatModel"]
    credential: CredentialPolicy = field(default_factory=CredentialPolicy)

    def build(self, request: ModelRequest) -> "BaseChatModel":
        """Prepare the request and construct the client."""
        prepared = self.prepare(request)
        return self.protocol_build(prepared.client_kwargs)


def is_local_http_endpoint(base_url: str | None) -> bool:
    """Return True for local HTTP endpoints such as an ``argo-shim``.

    Moved verbatim from ``models/openai.py`` so endpoint modules share one
    definition.
    """
    if not base_url:
        return False
    parsed = urlparse(base_url)
    return parsed.scheme == "http" and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "::1",
        "0.0.0.0",
    }


def resolve_api_key(
    policy: CredentialPolicy,
    explicit: str | None,
    *,
    persist_env: bool = True,
) -> str | None:
    """Resolve an API key following an endpoint's credential policy.

    Order: explicit value -> environment variable -> interactive prompt (only
    when attached to a TTY) -> placeholder. Raises ``ValueError`` when a
    required key cannot be resolved and no prompt is possible.
    """
    if explicit:
        return explicit

    key = os.getenv(policy.env_var) if policy.env_var else None
    if key:
        return key

    if policy.interactive_prompt and sys.stdin.isatty():
        if policy.env_var:
            logger.info("%s not found in environment variables.", policy.env_var)
        key = getpass(_prompt_text(policy))
        if key and persist_env and policy.env_var:
            os.environ[policy.env_var] = key
        return key

    if policy.placeholder is not None:
        return policy.placeholder

    if policy.required:
        raise ValueError(policy.missing_key_help or _default_missing_help(policy))

    return None


def _prompt_text(policy: CredentialPolicy) -> str:
    if policy.env_var:
        return f"Please enter your {policy.env_var} value: "
    return "Please enter your API key: "


def _default_missing_help(policy: CredentialPolicy) -> str:
    if policy.env_var:
        return (
            f"API key not found. Set the {policy.env_var} environment variable."
        )
    return "API key not found."
