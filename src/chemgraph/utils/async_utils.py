"""Async helpers shared across ChemGraph CLI and UI."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Callable


def run_async_callable(fn: Callable[..., Any]) -> Any:
    """Run an async callable and return its result in a sync context.

    If no event loop is running, uses ``asyncio.run`` directly.
    Otherwise, spawns a daemon thread so that the call does not
    conflict with an already-running loop (e.g. inside Streamlit).
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(fn())

    result_container: dict[str, Any] = {}
    error_container: dict[str, Exception] = {}

    def runner() -> None:
        try:
            result_container["value"] = asyncio.run(fn())
        except Exception as exc:
            error_container["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in error_container:
        raise error_container["error"]
    return result_container.get("value")
