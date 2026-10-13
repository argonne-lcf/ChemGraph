"""Durable outbox and credential-isolated Hermes delivery helper."""

import json
import time

from starlette.responses import JSONResponse
from starlette.routing import Route


def notification_routes(service):
    async def claim(request):
        async with service.lock:
            async with service.store.transaction():
                # Timed-out deliveries need human reconciliation, never a blind retry.
                await service.store.db.execute(
                    "UPDATE outbox SET state='ambiguous' "
                    "WHERE state='delivering' AND claimed_at < ?",
                    (time.time() - 300,),
                )
                row = await service.store.one(
                    "SELECT * FROM outbox WHERE state='pending' "
                    "AND next_attempt<=? ORDER BY rowid LIMIT 1",
                    (time.time(),),
                )
                if not row:
                    return JSONResponse({"notification": None})
                await service.store.db.execute(
                    "UPDATE outbox SET state='delivering', "
                    "claimed_at=?, attempts=attempts+1 WHERE id=?",
                    (time.time(), row["id"]),
                )
        return JSONResponse({"notification": json.loads(row["payload"])})

    async def acknowledge(request):
        data = await request.json()
        state = data.get("state")
        if state not in {"delivered", "retry", "ambiguous"}:
            return JSONResponse({"error": "Invalid delivery state"}, status_code=400)
        async with service.lock:
            row = await service.store.one("SELECT * FROM outbox WHERE id=?", (data.get("id"),))
            if not row or row["state"] != "delivering":
                return JSONResponse({"error": "No active delivery"}, status_code=409)
            async with service.store.transaction():
                await service.store.db.execute(
                    "UPDATE outbox SET state=?,receipt=?,next_attempt=? WHERE id=?",
                    (
                        "pending" if state == "retry" else state,
                        str(data.get("receipt", ""))[:4000],
                        time.time() + min(3600, 2 ** min(row["attempts"], 12)),
                        row["id"],
                    ),
                )
        return JSONResponse({"ok": True})

    async def inspect(request):
        rows = await service.store.all("SELECT * FROM outbox ORDER BY rowid DESC LIMIT 100")
        return JSONResponse([dict(row) for row in rows])

    async def resolve(request):
        data = await request.json()
        if data.get("state") not in {"delivered", "pending"} or not data.get("evidence"):
            return JSONResponse(
                {"error": "Choose delivered or pending and supply reviewed evidence"},
                status_code=400,
            )
        async with service.lock:
            row = await service.store.one(
                "SELECT state FROM outbox WHERE id=?", (request.path_params["identifier"],)
            )
            if not row or row["state"] != "ambiguous":
                return JSONResponse({"error": "Notification is not ambiguous"}, status_code=409)
            async with service.store.transaction():
                await service.store.db.execute(
                    "UPDATE outbox SET state=?,next_attempt=0,receipt=? WHERE id=?",
                    (
                        data["state"],
                        json.dumps(
                            {"operator": request.user.display_name, "evidence": data["evidence"]}
                        ),
                        request.path_params["identifier"],
                    ),
                )
        return JSONResponse({"ok": True})

    return [
        Route("/notifications/claim", claim, methods=["POST"]),
        Route("/notifications/ack", acknowledge, methods=["POST"]),
        Route("/admin/notifications", inspect),
        Route("/admin/notifications/{identifier}/resolve", resolve, methods=["POST"]),
    ]
