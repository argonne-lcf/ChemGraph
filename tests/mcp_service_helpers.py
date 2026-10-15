"""Shared hermetic service fixtures and JSON-RPC helpers."""

import asyncio
from contextlib import asynccontextmanager
import json

import httpx

from chemgraph.mcp.service.config import ServiceConfig

WATER = "3\nwater\nO 0 0 0\nH 0 0 0.96\nH 0.93 0 -0.24\n"


def config(tmp_path, **kwargs):
    return ServiceConfig(data_dir=tmp_path, public_url="http://localhost", enable_pubchem=False, **kwargs)


@asynccontextmanager
async def running_app(settings):
    from chemgraph.mcp.service.app import create_app

    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=settings.public_url) as client:
            yield app, client


def headers(token):
    return {"Authorization": f"Bearer {token}", "Accept": "application/json, text/event-stream"}


async def rpc(client, token, method, params):
    response = await client.post("/mcp", headers=headers(token),
                                 json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    assert response.status_code == 200, response.text
    return response.json()["result"]


async def tool(client, token, name, **arguments):
    result = await rpc(client, token, "tools/call", {"name": name, "arguments": arguments})
    assert not result.get("isError"), result
    if "structuredContent" in result:
        content = result["structuredContent"]
        return content.get("result", content) if set(content) == {"result"} else content
    return json.loads(result["content"][0]["text"])


async def completed(client, token, identifier, timeout=90):
    async with asyncio.timeout(timeout):
        while True:
            state = await tool(client, token, "check_job_status", batch_id=identifier)
            if state["status"] not in {"approved", "invoking", "returned"}:
                return state
            await asyncio.sleep(0.1)
