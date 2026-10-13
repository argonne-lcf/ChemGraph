"""Optional Academy adapter. No graph, tool, or conversation policy lives here."""

from __future__ import annotations

import asyncio
from typing import Any

from academy.agent import Agent, action
from academy.handle import Handle
from academy.runtime import Runtime, RuntimeConfig
from academy.serialize import SerializationStrategy

from chemgraph.distributed.protocol import PeerEnvelope, Receiver, Status


class PeerMailbox(Agent):
    def __init__(self, receive: Receiver, status: Status):
        super().__init__()
        self.receive, self.status = receive, status

    @action
    async def receive_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.receive(PeerEnvelope.model_validate(payload))

    @action
    async def get_status(self) -> dict[str, Any]:
        return self.status()


class AcademyTransport:
    """Public Academy registration/discovery with a bounded delivery acknowledgement.

    Share a LocalExchangeFactory between adapters for in-process tests; use an
    HttpExchangeFactory for independently launched peers. Every launch must have
    a fresh launch_id, shared by its peers and operator clients.
    """

    def __init__(self, *, exchange_factory: Any, run_id: str, launch_id: str,
                 name: str | None = None, timeout_s: float = 30):
        self.factory, self.run_id, self.launch_id = exchange_factory, run_id, launch_id
        self.name, self.timeout_s = name, timeout_s
        self._client = None
        self._runtime: Runtime | None = None
        self._task: asyncio.Task | None = None
        self._registration = None

    async def __aenter__(self):
        self._client = await self.factory.create_user_client(name=f"chemgraph/{self.run_id}/{self.launch_id}/client")
        return self

    async def __aexit__(self, *_args):
        await self.close()

    def _peer_name(self, name: str) -> str:
        return f"chemgraph/{self.run_id}/{self.launch_id}/{name}"

    async def start(self, receive: Receiver, status: Status) -> None:
        if self._client is None or self.name is None or self._task is not None:
            raise RuntimeError("Enter an unused named AcademyTransport before starting it")
        self._registration = await self._client.register_agent(PeerMailbox, name=self._peer_name(self.name))
        ready = asyncio.get_running_loop().create_future()

        def identify():
            return {**status(), "run_id": self.run_id, "launch_id": self.launch_id, "peer": self.name}

        async def serve():
            self._runtime = Runtime(
                PeerMailbox(receive, identify), exchange_factory=self.factory,
                registration=self._registration, config=RuntimeConfig(
                    default_serializer=SerializationStrategy.JSON,
                    allowed_deserializers={SerializationStrategy.JSON},
                ),
            )
            try:
                async with self._runtime:
                    ready.set_result(None)
                    await self._runtime.wait_shutdown()
            except BaseException as exc:
                if not ready.done():
                    ready.set_exception(exc)
                raise

        self._task = asyncio.create_task(serve())
        await ready

    async def _find_peer(self, name: str) -> Handle:
        if self._client is None:
            raise RuntimeError("Academy transport is closed")
        while True:
            # HTTP discovery returns UUIDs without display names. Resolve the
            # ChemGraph identity through the public mailbox action instead.
            candidates = [aid for aid in await self._client.discover(PeerMailbox)
                          if aid.name is None or aid.name == self._peer_name(name)]

            async def probe(aid):
                handle = Handle(aid, exchange=self._client, ignore_context=True,
                                request_serializer=SerializationStrategy.JSON,
                                result_serializer=SerializationStrategy.JSON,
                                exception_serializer=SerializationStrategy.JSON)
                try:
                    state = await asyncio.wait_for(handle.action("get_status"), min(1, self.timeout_s))
                except Exception:
                    return None  # An abandoned mailbox is not a live peer.
                if (state.get("run_id"), state.get("launch_id"), state.get("peer")) != (self.run_id, self.launch_id, name):
                    return None
                if state.get("state") not in {"starting", "stopping", "stopped"}:
                    return handle
                return None

            ready = [handle for handle in await asyncio.gather(*(probe(aid) for aid in candidates)) if handle is not None]
            if len(ready) > 1:
                raise RuntimeError(f"Multiple live registrations for peer {name!r}; stop the duplicate process")
            if ready:
                return ready[0]
            await asyncio.sleep(0.1)

    async def peer_status(self, name: str) -> dict:
        async with asyncio.timeout(self.timeout_s):
            handle = await self._find_peer(name)
            return await handle.action("get_status")

    def check_health(self) -> None:
        if self._task is not None and self._task.done():
            self._task.result()
            raise RuntimeError("Academy mailbox stopped")

    async def send(self, message: PeerEnvelope) -> dict[str, Any]:
        self.check_health()
        if (message.run_id, message.launch_id) != (self.run_id, self.launch_id):
            raise ValueError("Message belongs to a different launch")
        expected_sender = self.name or "operator"
        if message.sender != expected_sender:
            raise ValueError("Message sender does not match this transport")
        try:
            async with asyncio.timeout(self.timeout_s):
                handle = await self._find_peer(message.recipient)
                result = await handle.action("receive_message", message.model_dump(mode="json"))
                if result != {"status": "accepted", "message_id": message.message_id}:
                    raise RuntimeError("Peer returned an invalid delivery acknowledgement")
                return result
        except TimeoutError as exc:
            raise TimeoutError(
                f"No acknowledgement for message {message.message_id} to {message.recipient!r}; "
                "delivery may have occurred. Inspect peer status before resubmitting."
            ) from exc

    async def close(self) -> None:
        try:
            if self._task is not None:
                if self._runtime is not None:
                    self._runtime.signal_shutdown()
                await self._task
        finally:
            if self._client is not None:
                await self._client.close()
                self._client = None
