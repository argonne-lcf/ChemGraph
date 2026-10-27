"""One Academy action and one sequential inbox consumer."""

import asyncio
from collections.abc import Awaitable, Callable

from academy.agent import Agent, action, loop

from chemgraph.academy.message import PeerMessage


class ChemGraphPeer(Agent):
    """Keep a caller-owned asynchronous workflow available between turns.

    ``on_turn`` must return only after approvals and the turn are resolved.
    On failure, ``on_error`` must report the error and recover or replace the
    workflow before returning. It may wait for operator input; reception
    continues while it waits. If it raises, Academy's loop-error policy applies.

    Acceptance is not execution success. The inbox and seen message IDs live
    only in this instance, consume memory until shutdown, and are not replayed.
    Academy cancels the consumer on shutdown; handlers must allow cancellation.
    """

    def __init__(
        self,
        on_turn: Callable[[PeerMessage], Awaitable[None]],
        *,
        on_error: Callable[[PeerMessage, Exception], Awaitable[None]],
    ):
        super().__init__()
        self.on_turn = on_turn
        self.on_error = on_error
        self._inbox: asyncio.Queue[PeerMessage] = asyncio.Queue()
        self._seen: set[tuple[str, str]] = set()
        self._closed = False

    @action
    async def receive_message(self, payload: dict) -> dict:
        """Acknowledge inbox acceptance, including previously accepted IDs."""
        if self._closed:
            raise RuntimeError("Peer is shut down")
        message = PeerMessage.model_validate(payload)
        identity = (message.sender, message.message_id)
        if identity not in self._seen:
            self._inbox.put_nowait(message)
            self._seen.add(identity)
        return {"status": "accepted", "message_id": message.message_id}

    @loop
    async def consume_inbox(self, shutdown: asyncio.Event) -> None:
        while not shutdown.is_set():
            message = await self._inbox.get()
            try:
                try:
                    await self.on_turn(message)
                except Exception as error:
                    await self.on_error(message, error)
            finally:
                self._inbox.task_done()

    async def agent_on_shutdown(self) -> None:
        self._closed = True
        while not self._inbox.empty():
            self._inbox.get_nowait()
            self._inbox.task_done()
        self._seen.clear()
