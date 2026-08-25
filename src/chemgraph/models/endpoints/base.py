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
import re
import sys
from dataclasses import dataclass, field, replace
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
    config_section: str | None = None
    legacy_config_sections: tuple[str, ...] = ()
    base_url_resolver: Callable[[str, str | None], str | None] | None = None
    config_url_applies: Callable[[str], bool] = field(
        default=lambda _model: True,
        repr=False,
        compare=False,
    )
    curated_models: tuple[str, ...] = ()
    accepted_prefix: str | None = None
    display_name: str | None = None
    model_type: str = "Cloud"
    supports_structured_output: bool = True

    def resolve_base_url(
        self,
        model: str,
        base_url: str | None,
        *,
        from_config: bool = False,
    ) -> str | None:
        """Resolve a URL using this endpoint's normalization and defaults."""
        if from_config and not self.config_url_applies(model):
            base_url = None
        if self.base_url_resolver is None:
            return base_url
        return self.base_url_resolver(model, base_url)

    def prepare_request(self, request: ModelRequest) -> PreparedModel:
        """Prepare a request and attach capabilities declared by the spec."""
        prepared = self.prepare(request)
        if prepared.supports_structured_output != self.supports_structured_output:
            prepared = replace(
                prepared,
                supports_structured_output=self.supports_structured_output,
            )
        return prepared

    def build(self, request: ModelRequest) -> "BaseChatModel":
        """Prepare the request and construct the client."""
        prepared = self.prepare_request(request)
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


def normalize_openai_base_url(base_url: str | None) -> str | None:
    """Normalize legacy Argo resource URLs to OpenAI-compatible roots."""
    if not base_url:
        return base_url
    if (
        "apps-dev.inside.anl.gov/argoapi" in base_url
        or "apps.inside.anl.gov/argoapi" in base_url
    ):
        base_url = re.sub(r"/api/v1/resource/(chat|embed)/?$", "/v1", base_url)
        base_url = re.sub(r"/docs/?$", "", base_url)
        base_url = re.sub(r"/api/v1/?$", "/v1", base_url)
        base_url = base_url.rstrip("/")
    return base_url


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


def missing_credential_help(policy: CredentialPolicy) -> str:
    """Return endpoint-provided or generic missing-credential guidance."""
    return policy.missing_key_help or _default_missing_help(policy)
