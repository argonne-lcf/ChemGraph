"""Locally issued bearer tokens and the outer ASGI authentication boundary."""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import hmac
import logging
import secrets
import time
import uuid

from starlette.requests import Request
from starlette.responses import JSONResponse

from chemgraph.runtime.sqlite_store import connection, initialize_database


@dataclass(frozen=True)
class Principal:
    user_id: str
    name: str
    token_id: str


_principal: ContextVar[Principal | None] = ContextVar("chemgraph_principal", default=None)
logger = logging.getLogger(__name__)


def principal():
    value = _principal.get()
    if value is None:
        raise PermissionError("Authentication required.")
    return value


class TokenStore:
    """Tokens contain 256 random secret bits; only a SHA-256 digest is stored."""

    def __init__(self, path):
        self.path = initialize_database(path)
        with connection(path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY, name TEXT UNIQUE NOT NULL)""")
            db.execute("""CREATE TABLE IF NOT EXISTS tokens (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id),
                digest TEXT NOT NULL, expires_at REAL NOT NULL, revoked INTEGER NOT NULL DEFAULT 0
            )""")

    @staticmethod
    def _new(db, user_id, lifetime):
        if lifetime <= 0:
            raise ValueError("Token lifetime must be positive.")
        identifier = uuid.uuid4().hex
        token = f"cg_{identifier}.{secrets.token_urlsafe(32)}"
        db.execute("INSERT INTO tokens(id, user_id, digest, expires_at) VALUES (?, ?, ?, ?)",
                   (identifier, user_id, hashlib.sha256(token.encode()).hexdigest(), time.time() + lifetime))
        return token

    def issue(self, name, *, lifetime=30 * 86400):
        if not name or len(name) > 128:
            raise ValueError("A user name of 1–128 characters is required.")
        with connection(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO users VALUES (?, ?)", (uuid.uuid4().hex, name))
            owner = db.execute("SELECT id FROM users WHERE name=?", (name,)).fetchone()[0]
            return self._new(db, owner, lifetime)

    def rotate(self, identifier, *, lifetime=30 * 86400):
        with connection(self.path) as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT user_id FROM tokens WHERE id=? AND revoked=0", (identifier,)).fetchone()
            if row is None:
                raise ValueError("Unknown active token ID.")
            token = self._new(db, row[0], lifetime)
            db.execute("UPDATE tokens SET revoked=1 WHERE id=?", (identifier,))
            return token

    def revoke(self, identifier):
        with connection(self.path) as db:
            return db.execute("UPDATE tokens SET revoked=1 WHERE id=?", (identifier,)).rowcount > 0

    def list_tokens(self):
        with connection(self.path) as db:
            return [dict(row) for row in db.execute("""SELECT tokens.id, user_id, users.name,
                expires_at, revoked FROM tokens JOIN users ON users.id=tokens.user_id ORDER BY tokens.rowid""")]

    def verify(self, token):
        if len(token) > 256 or not token.startswith("cg_") or "." not in token:
            return None
        identifier = token[3:].split(".", 1)[0]
        with connection(self.path) as db:
            row = db.execute("""SELECT tokens.*, users.name FROM tokens
                JOIN users ON users.id=tokens.user_id WHERE tokens.id=?""", (identifier,)).fetchone()
        if (row is None or row["revoked"] or row["expires_at"] <= time.time()
                or not hmac.compare_digest(row["digest"], hashlib.sha256(token.encode()).hexdigest())):
            return None
        return Principal(row["user_id"], row["name"], row["id"])


class BearerMiddleware:
    """Protect all application routes except the two deliberately public probes.

    Pure ASGI middleware keeps ContextVars intact in MCP's stateless task tree.
    No OAuth metadata or fictitious issuer is advertised for these local tokens.
    """

    def __init__(self, app, *, tokens, config):
        self.app, self.tokens, self.config = app, tokens, config

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        request = Request(scope)
        if request.headers.get("host", "").lower() != self.config.allowed_host:
            return await JSONResponse({"error": "Invalid host"}, status_code=421)(scope, receive, send)
        origin = request.headers.get("origin")
        if origin is not None and origin not in self.config.allowed_origins:
            return await JSONResponse({"error": "Invalid origin"}, status_code=403)(scope, receive, send)
        if scope["path"] in {"/healthz", "/readyz"} and scope["method"] in {"GET", "HEAD"}:
            return await self.app(scope, receive, send)
        header = request.headers.get("authorization", "")
        scheme, _, value = header.partition(" ")
        caller = await asyncio.to_thread(self.tokens.verify, value) if scheme.lower() == "bearer" else None
        if caller is None:
            response = JSONResponse({"error": "Authentication required"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            return await response(scope, receive, send)
        if scope["path"].rstrip("/") == "/mcp" and scope["method"] == "POST":
            # Bound chunked requests too, before handing JSON to the MCP parser.
            body = bytearray()
            async for chunk in Request(scope, receive).stream():
                if len(body) + len(chunk) > self.config.max_text_bytes * 8:
                    return await JSONResponse({"error": "Request too large"}, status_code=413)(scope, receive, send)
                body.extend(chunk)
            original_receive = receive
            consumed = False

            async def replay():
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await original_receive()

            receive = replay
        token = _principal.set(caller)
        scope["chemgraph.principal"] = caller
        request_id = uuid.uuid4().hex
        status = None

        async def logged_send(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, logged_send)
        finally:
            _principal.reset(token)
            logger.info("service_request request=%s owner=%s method=%s status=%s",
                        request_id, caller.user_id, scope["method"], status)
