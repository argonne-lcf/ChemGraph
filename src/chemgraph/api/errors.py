"""Safe public errors. Provider exception strings belong only in protected logs."""


class RunStopped(Exception):
    pass


class StorageLimit(Exception):
    pass


def public_error(exc):
    if isinstance(exc, StorageLimit):
        return (
            "storage_limit",
            "Your workspace storage limit was reached. Contact your administrator.",
        )
    # Inspect typed status metadata, never provider response text.
    chain = []
    while exc is not None and id(exc) not in {id(item) for item in chain}:
        chain.append(exc)
        exc = exc.__cause__ or exc.__context__
    for error in chain:
        code = getattr(error, "status_code", None) or getattr(error, "code", None)
        name = type(error).__name__
        if code in {401, 403} or name in {
            "AuthenticationError",
            "PermissionDeniedError",
            "Unauthenticated",
            "PermissionDenied",
        }:
            return (
                "provider_authentication",
                "The provider rejected the shared credentials. Ask your administrator to renew them.",
            )
        if code == 429 or name in {
            "RateLimitError",
            "ResourceExhausted",
            "TooManyRequests",
        }:
            return (
                "provider_rate_limit",
                "The provider's rate limit was reached. Wait before explicitly retrying this run.",
            )
        if (
            code in {408, 504}
            or isinstance(error, TimeoutError)
            or "Timeout" in name
            or name == "DeadlineExceeded"
        ):
            return (
                "provider_timeout",
                "The provider did not respond in time. You can retry this run.",
            )
        if isinstance(error, ConnectionError) or name in {
            "APIConnectionError",
            "ConnectError",
            "NetworkError",
            "ServiceUnavailable",
        }:
            return (
                "provider_connection",
                "The provider could not be reached. Check with your administrator or retry later.",
            )
    return (
        "calculation_failed",
        "The calculation failed. Contact your administrator with this run ID.",
    )
