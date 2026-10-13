"""Polling client shared by the CLI and optional Hermes extension."""

from uuid import uuid4

import httpx
from a2a.client.transports.jsonrpc import JsonRpcTransport
from a2a.types import a2a_pb2 as a2a
from google.protobuf.json_format import ParseDict


class Client:
    def __init__(self, url, token, transport=None):
        self.url = url.rstrip("/")
        self.http = httpx.AsyncClient(
            base_url=self.url,
            headers={"Authorization": f"Bearer {token}", "A2A-Version": "1.0"},
            timeout=60,
            transport=transport,
        )

    async def __aenter__(self):
        response = await self.http.get("/.well-known/agent-card.json")
        response.raise_for_status()
        card = ParseDict(response.json(), a2a.AgentCard())
        # Use the explicitly configured origin, not an untrusted card URL.
        self.rpc = JsonRpcTransport(self.http, card, self.url + "/")
        return self

    async def __aexit__(self, *args):
        await self.http.aclose()

    async def send(
        self, text, *, context_id="", task_id="", message_id=None, blocking=False, interrupt_id=None
    ):
        message = a2a.Message(
            message_id=message_id or str(uuid4()),
            role=a2a.ROLE_USER,
            context_id=context_id,
            task_id=task_id,
            parts=[a2a.Part(text=text)],
        )
        if interrupt_id:
            message.metadata.update({"interrupt_id": interrupt_id})
        return await self.rpc.send_message(
            a2a.SendMessageRequest(
                message=message,
                configuration=a2a.SendMessageConfiguration(return_immediately=not blocking),
            )
        )

    async def get(self, task_id):
        return await self.rpc.get_task(a2a.GetTaskRequest(id=task_id))

    async def cancel(self, task_id):
        return await self.rpc.cancel_task(a2a.CancelTaskRequest(id=task_id))

    async def list(self, context_id=""):
        return await self.rpc.list_tasks(a2a.ListTasksRequest(context_id=context_id))
