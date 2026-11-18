"""Invocation-local side channel; accounting evidence never enters messages.

These hooks target the pinned LangChain v1 chat-completion paths. Responses API,
native v2 streaming, and cache hits that bypass the hooks remain unverified.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)
_active = ContextVar("chemgraph_usage_capture", default=None)


def mapping(value):
    """Retain fields actually supplied by the provider, including unknown keys."""
    if isinstance(value, dict):
        return deepcopy(value)
    if callable(getattr(value, "model_dump", None)):
        return value.model_dump(mode="json", exclude_unset=True)
    return {}


def merge_usage(previous, update):
    """Merge cumulative snapshots, never add them or erase counts with nulls."""
    result = deepcopy(previous) if isinstance(previous, dict) else {}
    for key, value in update.items():
        if isinstance(value, dict):
            result[key] = merge_usage(result.get(key) or {}, value)
        elif value is not None or key not in result:
            result[key] = value
    return result


class _Capture:
    def __init__(self, run_manager=None):
        self.run_id = getattr(run_manager, "run_id", None)
        self.handlers = [h for h in getattr(run_manager, "handlers", [])
                         if callable(getattr(h, "on_usage_evidence", None))]
        self.usage = {}


@contextmanager
def _scope(capture):
    token = _active.set(capture)
    try:
        yield
    finally:
        _active.reset(token)


def bind_collector(collector, run_id):
    """Public stream() does not pass a run manager to the provider hook."""
    capture = _active.get()
    if capture is not None and capture.run_id in (None, run_id):
        capture.run_id = run_id
        if collector not in capture.handlers:
            capture.handlers.append(collector)


def capture_usage(value, *, provider, model=None, event="response", complete=False):
    capture = _active.get()
    if capture is None or not capture.handlers:
        return
    try:
        raw = mapping(value)
        # An earlier explicit total is stale if a later snapshot changes a
        # component without supplying a new total. Normalize from components.
        for total, components in (
            ("total_tokens", ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens",
                              "cache_read_input_tokens", "cache_creation_input_tokens")),
            ("totalTokens", ("inputTokens", "outputTokens")),
        ):
            if raw.get(total) is None and any(
                raw.get(key) is not None and raw[key] != capture.usage.get(key)
                for key in components
            ):
                capture.usage.pop(total, None)
        capture.usage = merge_usage(capture.usage, raw)
        for handler in capture.handlers:
            handler.on_usage_evidence(
                capture.run_id, deepcopy(capture.usage), provider=provider,
                model=model, event=event, raw=deepcopy(raw), complete=complete,
            )
    except Exception:
        # Accounting must not change model behavior, including exception paths.
        logger.warning("Could not capture provider usage.", exc_info=True)


class UsageCaptureMixin:
    """Scope hooks to one run, including batched and interleaved streams."""

    def _generate_with_cache(self, messages, stop=None, run_manager=None, **kwargs):
        with _scope(_Capture(run_manager)):
            return super()._generate_with_cache(messages, stop, run_manager, **kwargs)

    async def _agenerate_with_cache(self, messages, stop=None, run_manager=None, **kwargs):
        with _scope(_Capture(run_manager)):
            return await super()._agenerate_with_cache(messages, stop, run_manager, **kwargs)

    def stream(self, *args, **kwargs):
        capture = _Capture()
        iterator = super().stream(*args, **kwargs)
        try:
            while True:
                with _scope(capture):
                    try:
                        chunk = next(iterator)
                    except StopIteration:
                        return
                # Reset before yielding: callers may interleave streams on a thread.
                yield chunk
        finally:
            with _scope(capture):
                iterator.close()

    async def astream(self, *args, **kwargs):
        capture = _Capture()
        iterator = super().astream(*args, **kwargs)
        try:
            while True:
                with _scope(capture):
                    try:
                        chunk = await anext(iterator)
                    except StopAsyncIteration:
                        return
                yield chunk
        finally:
            with _scope(capture):
                await iterator.aclose()
