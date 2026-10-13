"""Inbox processing and lifecycle policy owned by ChemGraph."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Callable
from typing import Any

from pydantic_core import to_jsonable_python

from chemgraph.distributed.protocol import PeerEnvelope, PeerRunner, PeerTurnResult


class PeerRuntime:
    def __init__(self, *, name: str, run_id: str, launch_id: str,
                 allowed_senders: tuple[str, ...], runner: PeerRunner,
                 trace: Callable[[str, dict], None] | None = None):
        self.name, self.run_id, self.launch_id = name, run_id, launch_id
        self.allowed_senders = allowed_senders
        self.runner = runner
        self.trace = trace or (lambda *_args: None)
        self.inbox: asyncio.Queue[PeerEnvelope] = asyncio.Queue()
        self.state = "starting"
        self.current: PeerEnvelope | None = None
        self.output = ""
        self.last_error: str | None = None
        self.last_activity = time.monotonic()
        self._continue = asyncio.Event()
        self._control_lock = asyncio.Lock()
        self._worker: asyncio.Task | None = None
        self._closing = False
        self._interrupts: tuple[dict, ...] = ()
        self.instance_id = str(uuid.uuid4())
        # Retain acceptance records for this process's lifetime, including completed turns.
        self._accepted: dict[tuple[str, str], str] = {}
        self._control_epoch = str(uuid.uuid4())

    async def start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("Peer runtime already started")
        self.state = "failed" if getattr(self.runner, "failed", False) else "waiting_for_user" if self.runner.interrupts else "idle"
        if self.state == "idle":
            self._continue.set()
        self._worker = asyncio.create_task(self._process())

    async def receive(self, message: PeerEnvelope) -> dict[str, Any]:
        if self._closing or self._worker is None or self._worker.done():
            raise RuntimeError("Peer is not accepting messages")
        if (message.run_id, message.launch_id, message.recipient) != (self.run_id, self.launch_id, self.name):
            raise ValueError("Message does not belong to this peer launch")
        if message.sender != "operator" and message.sender not in self.allowed_senders:
            raise ValueError(f"Unexpected sender: {message.sender!r}")
        key = (message.sender, message.message_id)
        fingerprint = hashlib.sha256(message.model_dump_json().encode()).hexdigest()
        acknowledgement = {"status": "accepted", "message_id": message.message_id}
        if key in self._accepted:
            if self._accepted[key] != fingerprint:
                raise ValueError("Message ID was already accepted with different content")
            return acknowledgement
        if self.state == "failed":
            recovery = ("retry the failed turn" if getattr(self.runner, "supports_retry", False)
                        else "restart the failed peer")
            raise RuntimeError(f"Peer is failed; {recovery} before sending new work")
        self.inbox.put_nowait(message.model_copy(deep=True))
        self._accepted[key] = fingerprint
        self.last_activity = time.monotonic()
        self.trace("message_received", message.model_dump())
        return acknowledgement

    def status(self) -> dict[str, Any]:
        interrupts = self.runner.interrupts or self._interrupts
        state = self.state
        if interrupts and state not in {"failed", "resuming", "stopping", "stopped"}:
            state = "waiting_for_user"
        if self._closing and state != "stopped":
            state = "stopping"
        control_token = None
        if state in {"waiting_for_user", "failed"}:
            # The runner token distinguishes identical successive legacy questions.
            identity = f"{self._control_epoch}/{getattr(self.runner, 'interrupt_token', None)}"
            control_token = hashlib.sha256(identity.encode()).hexdigest()
        return {"peer": self.name, "instance_id": self.instance_id, "state": state,
                "pending_messages": self.inbox.qsize(), "output": self.output,
                "current_message_id": self.current.message_id if self.current else None,
                "last_error": self.last_error, "control_token": control_token,
                "interrupts": to_jsonable_python(interrupts, fallback=str)}

    def _record(self, result: PeerTurnResult) -> None:
        self._control_epoch = str(uuid.uuid4())
        self.output, self._interrupts = result.output, result.interrupts
        self.last_error = None
        self.state = "idle" if result.status == "completed" else result.status
        self.last_activity = time.monotonic()
        self.trace("workflow_finished", {"message_id": self.current.message_id if self.current else None,
                                        "status": result.status, "output": result.output})
        if result.status == "completed":
            self._continue.set()

    def _failed(self, exc: Exception) -> None:
        self._control_epoch = str(uuid.uuid4())
        self.last_error = str(exc)
        self.state = "failed"
        self.trace("agent_error", {"message_id": self.current.message_id if self.current else None, "error": str(exc)})

    async def _process(self) -> None:
        await self._continue.wait()
        while True:
            self.current = await self.inbox.get()
            self._continue.clear()
            self.state = "running"
            try:
                try:
                    self._record(await self.runner.run(self.current))
                except Exception as exc:
                    self._failed(exc)
                await self._continue.wait()
            finally:
                self.current = None
                self.inbox.task_done()

    def _consume_control(self, token: str | None) -> None:
        if token is not None and token != self.status()["control_token"]:
            raise RuntimeError("Peer state changed; inspect status and submit a new control request")
        self._control_epoch = str(uuid.uuid4())

    async def resume(self, response: Any, *, control_token: str | None = None) -> None:
        """Explicit operator response; peer messages never call this method."""
        async with self._control_lock:
            if self._closing or self.status()["state"] != "waiting_for_user":
                raise RuntimeError("Peer is not waiting for operator input")
            self._consume_control(control_token)
            self.state = "resuming"
            try:
                result = await self.runner.resume(response)
            except Exception as exc:
                if getattr(self.runner, "failed", False):
                    self._failed(exc)
                else:
                    self.state = "waiting_for_user"
                raise
            if result is not None:
                self._record(result)
            else:
                self.state = "running"

    async def retry(self, *, control_token: str | None = None) -> None:
        async with self._control_lock:
            if self._closing or self.state != "failed":
                raise RuntimeError("Peer has no failed turn to retry")
            self._consume_control(control_token)
            self.state = "running"
            try:
                self._record(await self.runner.retry())
            except Exception as exc:
                self._failed(exc)
                raise

    async def close(self, *, timeout_s: float = 30) -> None:
        self._closing = True
        self.state = "stopping"
        if self._worker is None:
            self.state = "stopped"
            return
        try:
            await asyncio.wait_for(self.inbox.join(), timeout_s)
        except asyncio.TimeoutError:
            self.trace("agent_error", {"error": "shutdown_timeout", "pending_messages": self.inbox.qsize(),
                                       "current_message_id": self.current.message_id if self.current else None})
        finally:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self.state = "stopped"
