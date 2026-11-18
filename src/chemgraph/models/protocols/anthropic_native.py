"""The only production construction site for ``ChatAnthropic``."""

from __future__ import annotations

from typing import Any

from langchain_anthropic import ChatAnthropic

from .usage_capture import UsageCaptureMixin, capture_usage


class UsageAwareChatAnthropic(UsageCaptureMixin, ChatAnthropic):
    """Retain message_start input counts when later deltas only report output."""

    def _format_output(self, data, **kwargs):
        capture_usage(data.usage, provider="anthropic", model=getattr(data, "model", None), complete=True)
        return super()._format_output(data, **kwargs)

    def _make_message_chunk_from_anthropic_event(self, event, **kwargs):
        if event.type == "message_start":
            capture_usage(event.message.usage, provider="anthropic",
                          model=getattr(event.message, "model", None), event=event.type)
        elif event.type == "message_delta":
            capture_usage(event.usage, provider="anthropic", event=event.type)
        elif event.type == "message_stop":
            capture_usage({}, provider="anthropic", event=event.type, complete=True)
        return super()._make_message_chunk_from_anthropic_event(event, **kwargs)


def build(client_kwargs: dict[str, Any]) -> ChatAnthropic:
    """Construct a ``ChatAnthropic`` from prepared keyword arguments."""
    return UsageAwareChatAnthropic(**client_kwargs)
