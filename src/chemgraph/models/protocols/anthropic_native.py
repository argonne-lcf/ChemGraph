"""The only production construction site for ``ChatAnthropic``."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import LanguageModelInput


class CachingChatAnthropic(ChatAnthropic):
    """``ChatAnthropic`` that caches the tools and system-prompt prefix."""

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        system = payload.get("system")
        if isinstance(system, str) and system:
            payload["system"] = [
                {
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                },
            ]
        return payload


def build(client_kwargs: dict[str, Any]) -> ChatAnthropic:
    """Construct a ``ChatAnthropic`` from prepared keyword arguments."""
    return CachingChatAnthropic(**client_kwargs)
