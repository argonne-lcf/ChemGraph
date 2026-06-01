"""Token-bucket rate limiter for LLM API calls.

Academy is LLM-agnostic, so rate limiting must be handled at the
ChemGraph layer.  This module provides a shared :class:`RateLimiter`
that agents ``await`` before each LLM call to stay within per-provider
API quotas.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class _ProviderBucket:
    """Token bucket state for a single LLM provider."""

    rpm: float
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        # Start with a full bucket.
        self.tokens = self.rpm


class RateLimiter:
    """Async token-bucket rate limiter keyed by LLM provider.

    Parameters
    ----------
    default_rpm : int
        Default requests-per-minute for providers not explicitly
        configured (default ``60``).
    provider_rpm : dict[str, int] or None
        Per-provider overrides.  Keys are provider prefixes or model
        names (e.g. ``"openai"``, ``"anthropic"``, ``"gpt-4o"``).

    Usage
    -----
    ::

        limiter = RateLimiter(default_rpm=60, provider_rpm={"openai": 500})
        await limiter.acquire("gpt-4o")  # blocks if bucket empty
    """

    # Map model-name prefixes to canonical provider keys so that
    # ``acquire("gpt-4o")`` matches a rule set for ``"openai"``.
    _PREFIX_MAP: dict[str, str] = {
        "gpt-": "openai",
        "o1": "openai",
        "o3": "openai",
        "o4": "openai",
        "argo:": "argo",
        "claude-": "anthropic",
        "gemini-": "google",
        "groq:": "groq",
        "llama": "alcf",
    }

    def __init__(
        self,
        default_rpm: int = 60,
        provider_rpm: dict[str, int] | None = None,
    ) -> None:
        self._default_rpm = default_rpm
        self._provider_rpm: dict[str, int] = provider_rpm or {}
        self._buckets: dict[str, _ProviderBucket] = {}

    def _resolve_provider(self, model_name: str) -> str:
        """Map a model name to a canonical provider key."""
        # Direct match first.
        if model_name in self._provider_rpm:
            return model_name

        # Prefix match.
        lower = model_name.lower()
        for prefix, provider in self._PREFIX_MAP.items():
            if lower.startswith(prefix):
                return provider

        return model_name

    def _get_bucket(self, provider: str) -> _ProviderBucket:
        """Get or create the bucket for *provider*."""
        if provider not in self._buckets:
            rpm = self._provider_rpm.get(provider, self._default_rpm)
            self._buckets[provider] = _ProviderBucket(rpm=rpm)
        return self._buckets[provider]

    async def acquire(self, model_name: str) -> None:
        """Wait until a request token is available for *model_name*.

        Refills the token bucket based on elapsed time, then consumes
        one token.  If the bucket is empty, sleeps until a token
        becomes available.
        """
        provider = self._resolve_provider(model_name)
        bucket = self._get_bucket(provider)

        async with bucket.lock:
            now = time.monotonic()
            elapsed = now - bucket.last_refill
            # Refill at rpm / 60 tokens per second.
            refill = elapsed * (bucket.rpm / 60.0)
            bucket.tokens = min(bucket.rpm, bucket.tokens + refill)
            bucket.last_refill = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return

            # Need to wait for a token.
            deficit = 1.0 - bucket.tokens
            wait_seconds = deficit / (bucket.rpm / 60.0)
            logger.debug(
                "Rate limit: waiting %.1fs for provider %s (rpm=%d)",
                wait_seconds,
                provider,
                bucket.rpm,
            )

        # Sleep outside the lock so other providers aren't blocked.
        await asyncio.sleep(wait_seconds)

        # Consume after waking.
        async with bucket.lock:
            bucket.tokens = 0.0
            bucket.last_refill = time.monotonic()
