"""The only production construction site for ``ChatOpenAI``.

Every OpenAI-compatible endpoint (OpenAI direct, both Argo variants, all three
ALCF clusters, OpenRouter, and vLLM/custom endpoints) reaches this builder. It
receives fully prepared kwargs and applies no endpoint-specific conditions.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI

from .usage_capture import UsageCaptureMixin, capture_usage


class UsageAwareChatOpenAI(UsageCaptureMixin, ChatOpenAI):
    """Observe provider usage before LangChain supplies missing counts as zero."""

    def _create_chat_result(self, response, generation_info=None):
        usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
        model = response.get("model") if isinstance(response, dict) else getattr(response, "model", None)
        capture_usage(usage, provider="openai", model=model, complete=True)
        return super()._create_chat_result(response, generation_info)

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        if chunk.get("usage") is not None:
            capture_usage(chunk["usage"], provider="openai", model=chunk.get("model"),
                          event="usage_chunk", complete=not chunk.get("choices") or all(
                              choice.get("finish_reason") for choice in chunk["choices"]
                          ))
        return super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info,
        )


def build(client_kwargs: dict[str, Any]) -> ChatOpenAI:
    """Construct a ``ChatOpenAI`` from prepared keyword arguments."""
    return UsageAwareChatOpenAI(**client_kwargs)
