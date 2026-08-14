"""Single-loop ownership for asynchronous local LangGraph checkpointers."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


DEFAULT_CHECKPOINT_DB = str(Path.home() / ".chemgraph" / "checkpoints.db")
_CLOSE_TIMEOUT_SECONDS = 5.0

logger = logging.getLogger(__name__)


class CheckpointRuntime:
    """Own one background event loop and its async SQLite connections."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._savers: dict[str, Any] = {}
        self._connections: dict[str, Any] = {}
        self._closed = False
        self._lifecycle_lock = threading.Lock()
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()

    @staticmethod
    def _normalize_path(db_path: str | None = None) -> str:
        return os.path.abspath(os.path.expanduser(db_path or DEFAULT_CHECKPOINT_DB))

    def run(self, operation: Callable[[], Any]) -> Any:
        """Execute one awaitable-producing callable on the owner loop."""
        async def invoke():
            return await operation()

        coroutine = invoke()
        with self._lifecycle_lock:
            if self._closed:
                coroutine.close()
                raise RuntimeError("Checkpoint runtime is closed.")
            try:
                future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
            except Exception:
                coroutine.close()
                raise
        try:
            return future.result()
        except BaseException:
            future.cancel()
            raise

    def open_sqlite(self, db_path: str | None = None):
        """Open or reuse a strict ``AsyncSqliteSaver`` on the owner loop."""
        path = self._normalize_path(db_path)

        async def open_saver():
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            if path in self._savers:
                return self._savers[path]
            parent = os.path.dirname(path) or os.curdir
            parent_existed = os.path.exists(parent)
            os.makedirs(parent, mode=0o700, exist_ok=True)
            default_parent = os.path.dirname(os.path.abspath(DEFAULT_CHECKPOINT_DB))
            if os.name == "posix" and (
                not parent_existed or parent == default_parent
            ):
                os.chmod(parent, 0o700)
            connection = await aiosqlite.connect(path)
            try:
                saver = AsyncSqliteSaver(
                    connection,
                    serde=JsonPlusSerializer(
                        pickle_fallback=False,
                        allowed_msgpack_modules=None,
                    ),
                )
                await saver.setup()
            except Exception:
                await connection.close()
                raise
            if os.name == "posix":
                os.chmod(path, 0o600)
                for suffix in ("-wal", "-shm"):
                    companion = path + suffix
                    if os.path.exists(companion):
                        os.chmod(companion, 0o600)
            self._connections[path] = connection
            self._savers[path] = saver
            return saver

        return self.run(open_saver)

    def close_sqlite(self, db_path: str | None = None) -> None:
        """Close and forget one cached SQLite saver connection."""
        path = self._normalize_path(db_path)

        async def close_saver():
            connection = self._connections.get(path)
            if connection is not None:
                await connection.close()
            self._connections.pop(path, None)
            self._savers.pop(path, None)

        self.run(close_saver)

    def delete_thread(self, saver: Any, thread_id: str) -> None:
        """Delete every checkpoint namespace belonging to a root thread."""
        self.run(lambda: saver.adelete_thread(thread_id))

    def close(self) -> None:
        """Close saver connections and stop the owner loop."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True

        async def shutdown():
            for path, connection in list(self._connections.items()):
                try:
                    await connection.close()
                except Exception:
                    logger.warning(
                        "Could not close checkpoint database %s", path, exc_info=True
                    )
            self._connections.clear()
            self._savers.clear()
            self._loop.call_soon(self._loop.stop)

        if threading.current_thread() is self._thread:
            self._loop.create_task(shutdown())
            return

        deadline = time.monotonic() + _CLOSE_TIMEOUT_SECONDS
        coroutine = shutdown()
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        except RuntimeError:
            coroutine.close()
            logger.warning("Checkpoint event loop stopped before shutdown completed.")
            return
        try:
            future.result(timeout=_CLOSE_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            logger.warning(
                "Checkpoint shutdown is still pending after %.1f seconds; "
                "the daemon owner thread will finish it in the background.",
                _CLOSE_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.warning("Checkpoint shutdown failed.", exc_info=True)
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except RuntimeError:
                pass

        remaining = max(0.0, deadline - time.monotonic())
        self._thread.join(timeout=remaining)
        if self._thread.is_alive():
            logger.warning("Checkpoint owner thread is still running after shutdown.")

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()


__all__ = ["CheckpointRuntime", "DEFAULT_CHECKPOINT_DB"]
