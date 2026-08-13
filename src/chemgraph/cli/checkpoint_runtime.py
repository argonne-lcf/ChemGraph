"""Single-loop ownership for asynchronous local LangGraph checkpointers."""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


DEFAULT_CHECKPOINT_DB = str(Path.home() / ".chemgraph" / "checkpoints.db")


class CheckpointRuntime:
    """Own one background event loop and its async SQLite connections."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._savers: dict[str, Any] = {}
        self._connections: dict[str, Any] = {}
        self._closed = False

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, operation: Callable[[], Any]) -> Any:
        """Execute one awaitable-producing callable on the owner loop."""
        if self._closed:
            raise RuntimeError("Checkpoint runtime is closed.")

        async def invoke():
            return await operation()

        return asyncio.run_coroutine_threadsafe(invoke(), self._loop).result()

    def open_sqlite(self, db_path: str | None = None):
        """Open or reuse a strict ``AsyncSqliteSaver`` on the owner loop."""
        path = os.path.abspath(os.path.expanduser(db_path or DEFAULT_CHECKPOINT_DB))
        if path in self._savers:
            return self._savers[path]

        async def open_saver():
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            parent = os.path.dirname(path) or os.curdir
            parent_existed = os.path.exists(parent)
            os.makedirs(parent, mode=0o700, exist_ok=True)
            default_parent = os.path.dirname(os.path.abspath(DEFAULT_CHECKPOINT_DB))
            if os.name == "posix" and (
                not parent_existed or parent == default_parent
            ):
                os.chmod(parent, 0o700)
            connection = await aiosqlite.connect(path)
            saver = AsyncSqliteSaver(
                connection,
                serde=JsonPlusSerializer(
                    pickle_fallback=False,
                    allowed_msgpack_modules=None,
                ),
            )
            await saver.setup()
            if os.name == "posix":
                os.chmod(path, 0o600)
                for suffix in ("-wal", "-shm"):
                    companion = path + suffix
                    if os.path.exists(companion):
                        os.chmod(companion, 0o600)
            self._connections[path] = connection
            self._savers[path] = saver
            return saver

        return asyncio.run_coroutine_threadsafe(open_saver(), self._loop).result()

    def delete_thread(self, saver: Any, thread_id: str) -> None:
        """Delete every checkpoint namespace belonging to a root thread."""
        self.run(lambda: saver.adelete_thread(thread_id))

    def close(self) -> None:
        """Close saver connections and stop the owner loop."""
        if self._closed:
            return

        async def close_connections():
            for connection in self._connections.values():
                await connection.close()

        asyncio.run_coroutine_threadsafe(close_connections(), self._loop).result()
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()


__all__ = ["CheckpointRuntime", "DEFAULT_CHECKPOINT_DB"]
