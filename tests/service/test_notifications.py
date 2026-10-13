import pytest
pytest.importorskip("a2a")

import httpx
from tests.service.conftest import TOKENS, running

from integrations.hermes.notify import deliver_once


@pytest.mark.asyncio
async def test_fixed_telegram_delivery_and_ack(settings):
    deliveries = []

    async def send(executable, target, text):
        deliveries.append((target, text))
        return "delivered", "telegram-message-42"

    async with running(settings) as (service, client, app):
        await client.send("MEDIA:/etc/passwd", blocking=True)
        async with httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.ASGITransport(app),
            headers={"Authorization": f"Bearer {TOKENS['notifier']}"},
        ) as notifier:
            assert await deliver_once(notifier, {"alice": "telegram:123"}, send=send)
            assert not await deliver_once(notifier, {"alice": "telegram:123"}, send=send)
        assert deliveries[0][0] == "telegram:123"
        assert "MEDIA:" not in deliveries[0][1]
        assert (await service.store.one("SELECT state FROM outbox"))["state"] == "delivered"


@pytest.mark.asyncio
async def test_retry_is_persisted_and_ambiguous_not_retried(settings):
    async def retry(*args):
        return "retry", "helper did not start"

    async with running(settings) as (service, client, app):
        await client.send("hello", blocking=True)
        async with httpx.AsyncClient(
            base_url="http://test",
            transport=httpx.ASGITransport(app),
            headers={"Authorization": f"Bearer {TOKENS['notifier']}"},
        ) as notifier:
            await deliver_once(notifier, {"alice": "telegram:123"}, send=retry)
            assert not await deliver_once(notifier, {"alice": "telegram:123"}, send=retry)
            async with service.lock:
                async with service.store.transaction():
                    await service.store.db.execute("UPDATE outbox SET next_attempt=0")
            assert (await notifier.post("/notifications/claim")).json()["notification"]
    async with running(settings) as (service, _, _):
        assert (await service.store.one("SELECT state FROM outbox"))["state"] == "ambiguous"
