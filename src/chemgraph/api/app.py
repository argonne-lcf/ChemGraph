"""Authenticated REST and SSE endpoints for the separately deployed React UI."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path, PureWindowsPath
import sqlite3
import time
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from pydantic import BaseModel, Field

from chemgraph.api.settings import Settings
from chemgraph.api.providers import model_status, provenance, session_status
from chemgraph.api.store import Store, TERMINAL
from chemgraph.api.worker import Supervisor


class SessionInput(BaseModel):
    model: str = Field(min_length=1, max_length=200)
    workflow: str = "single_agent"


class RunInput(BaseModel):
    query: str = Field(min_length=1, max_length=20000)
    request_id: UUID
    attachments: list[UUID] = Field(default_factory=list, max_length=10)


class HumanResponse(BaseModel):
    question_id: UUID
    answer: str = Field(min_length=1, max_length=20000)


def create_app(settings: Settings | None = None, *, start_workers=True) -> FastAPI:
    """Application factory; worker startup can be disabled in hermetic API tests."""
    settings = settings or Settings.from_env()
    model_status(settings)  # Invalid routes fail before creating storage/workers.
    store = Store(settings.data_dir)
    supervisor = Supervisor(store, settings)

    @asynccontextmanager
    async def lifespan(app):
        if start_workers:
            await supervisor.start()
        try:
            yield
        finally:
            if start_workers:
                await supervisor.stop()

    app = FastAPI(
        title="ChemGraph Web API",
        version="1",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.state.store = store
    app.state.settings = settings

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        detail = (
            exc.detail
            if isinstance(exc.detail, dict)
            else {"code": exc.status_code, "message": str(exc.detail)}
        )
        return JSONResponse(
            {"error": {**detail, "submission_rejected": True}},
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            {"error": {"code": 422, "message": "Invalid request fields."}},
            status_code=422,
        )

    @app.middleware("http")
    async def gateway_identity(request: Request, call_next):
        if request.url.path == "/healthz":
            return await call_next(request)
        identity = (
            settings.dev_user
            or request.headers.get(settings.identity_header, "").strip()
        )
        if not identity or len(identity) > 512:
            return JSONResponse(
                {
                    "error": {
                        "code": 401,
                        "message": "Sign in through your institution's gateway.",
                    }
                },
                status_code=401,
            )
        request.state.owner = hashlib.sha256(identity.encode()).hexdigest()
        request.state.identity = identity
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            # Browser mutations must be JSON/binary fetches from this origin.
            if request.headers.get("Origin") != settings.public_origin.rstrip("/"):
                return JSONResponse(
                    {
                        "error": {
                            "code": 403,
                            "message": "Request origin is not allowed.",
                        }
                    },
                    status_code=403,
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    def owned_session(request, session_id):
        row = store.one(
            "SELECT * FROM sessions WHERE id=? AND owner=?",
            (session_id, request.state.owner),
        )
        if row is None:
            raise HTTPException(404, "Conversation not found.")
        return row

    def owned_run(request, run_id):
        row = store.one(
            "SELECT r.* FROM runs r JOIN sessions s ON s.id=r.session_id WHERE r.id=? AND s.owner=?",
            (run_id, request.state.owner),
        )
        if row is None:
            raise HTTPException(404, "Run not found.")
        return row

    def owned_file(request, file_id):
        row = store.one(
            "SELECT f.* FROM files f JOIN sessions s ON s.id=f.session_id WHERE f.id=? AND s.owner=?",
            (file_id, request.state.owner),
        )
        if row is None:
            raise HTTPException(404, "Artifact not found.")
        try:
            path = store.file_path(row)
        except (ValueError, FileNotFoundError):
            raise HTTPException(404, "Artifact is no longer available.") from None
        return row, path

    def file_view(row):
        return {key: row[key] for key in ("id", "name", "kind", "size", "run_id")} | {
            "url": f"/api/v1/artifacts/{row['id']}/content",
            "preview_url": f"/api/v1/artifacts/{row['id']}/structure"
            if Path(row["name"]).suffix.lower() in {".xyz", ".pdb", ".cif", ".traj"}
            else None,
        }

    def run_view(row):
        data = {
            key: row[key]
            for key in (
                "id",
                "session_id",
                "query",
                "status",
                "final_text",
                "error",
                "error_code",
                "question_id",
                "question",
                "created",
            )
        }
        data["artifacts"] = [
            file_view(f)
            for f in store.rows("SELECT * FROM files WHERE run_id=?", (row["id"],))
        ]
        data["attachments"] = [
            file_view(f)
            for file_id in json.loads(row["attachments"])
            if (f := store.one("SELECT * FROM files WHERE id=?", (file_id,)))
        ]
        data["responses"] = [
            json.loads(e["data"])
            for e in store.rows(
                "SELECT data FROM events WHERE run_id=? AND type='human_response' ORDER BY id",
                (row["id"],),
            )
        ]
        return data

    @app.get("/healthz")
    def health():
        store.one("SELECT 1")
        if start_workers and (supervisor.task is None or supervisor.task.done()):
            raise HTTPException(503, "The run scheduler is unavailable.")
        return {"status": "ok"}

    @app.get("/api/v1/openapi.json")
    def schema():
        return app.openapi()

    @app.get("/api/v1/capabilities")
    def capabilities(request: Request):
        return {
            "user": request.state.identity,
            "models": ["demo"] if settings.demo else list(settings.providers),
            "model_status": model_status(settings),
            "workflows": ["single_agent", "multi_agent"],
            "calculators": settings.calculators,
            "upload_limit": settings.upload_limit,
            "demo": settings.demo,
        }

    @app.get("/api/v1/sessions")
    def sessions(request: Request):
        return store.rows(
            "SELECT id,model,workflow,title,created FROM sessions WHERE owner=? ORDER BY created DESC",
            (request.state.owner,),
        )

    @app.post("/api/v1/sessions", status_code=201)
    def create_session(data: SessionInput, request: Request):
        models = {"demo"} if settings.demo else settings.providers
        if data.model not in models or data.workflow not in {
            "single_agent",
            "multi_agent",
        }:
            raise HTTPException(422, "Choose an approved model and workflow.")
        readiness = model_status(settings)[data.model]
        if not readiness["configured"]:
            raise HTTPException(
                503, {"code": readiness["code"], "message": readiness["message"]}
            )
        session_id = str(uuid4())
        store.execute(
            "INSERT INTO sessions(id,owner,model,workflow,title,provenance,last_activity) VALUES (?,?,?,?,?,?,?)",
            (
                session_id,
                request.state.owner,
                data.model,
                data.workflow,
                "New conversation",
                provenance(settings, data.model),
                time.time(),
            ),
        )
        return {
            "id": session_id,
            "model": data.model,
            "workflow": data.workflow,
            "title": "New conversation",
        }

    @app.get("/api/v1/sessions/{session_id}")
    def session_detail(session_id: str, request: Request):
        session = owned_session(request, session_id)
        session.pop("owner")
        session["model_status"] = session_status(settings, session)
        session.pop("provenance")
        session["runs"] = [
            run_view(r)
            for r in store.rows(
                "SELECT * FROM runs WHERE session_id=? ORDER BY created", (session_id,)
            )
        ]
        return session

    @app.post("/api/v1/sessions/{session_id}/uploads", status_code=201)
    async def upload(session_id: str, request: Request, filename: str):
        session = owned_session(request, session_id)
        readiness = session_status(settings, session)
        if not readiness["configured"]:
            raise HTTPException(
                503, {"code": readiness["code"], "message": readiness["message"]}
            )
        name = Path(PureWindowsPath(filename).name).name
        if not name or len(name) > 200 or any(ord(c) < 32 for c in name):
            raise HTTPException(422, "Invalid filename.")
        if Path(name).suffix.lower() not in {
            ".xyz",
            ".pdb",
            ".cif",
            ".traj",
            ".json",
            ".csv",
            ".txt",
        }:
            raise HTTPException(
                422, "Supported uploads: XYZ, PDB, CIF, TRAJ, JSON, CSV, TXT."
            )
        reservation = str(uuid4())
        try:
            expected = int(request.headers.get("content-length", settings.upload_limit))
        except ValueError:
            raise HTTPException(422, "Invalid content length.") from None
        if expected < 0 or expected > settings.upload_limit:
            raise HTTPException(413, "This file exceeds the upload limit.")
        with store.transaction() as db:
            if db.execute(
                "SELECT 1 FROM runs WHERE session_id=? AND status IN ('queued','running','waiting_for_input','cancelling')",
                (session_id,),
            ).fetchone():
                raise HTTPException(
                    409, "Wait for the current run before attaching more files."
                )
            if (
                store.owner_usage(request.state.owner, db) + expected
                > settings.user_storage_limit
            ):
                raise HTTPException(
                    413,
                    {
                        "code": "storage_limit",
                        "message": "Your workspace storage limit has been reached.",
                    },
                )
            db.execute(
                "INSERT INTO upload_reservations VALUES (?,?,?)",
                (reservation, session_id, expected),
            )
        path = store.root / "staging" / reservation
        path.parent.mkdir(parents=True, exist_ok=True)
        size = 0
        try:
            with path.open("xb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > min(expected, settings.upload_limit):
                        raise HTTPException(413, "This file exceeds the upload limit.")
                    output.write(chunk)
            if not size:
                raise HTTPException(422, "The attachment is empty.")
            target = store.session_dir(session_id) / "uploads" / reservation / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with store.transaction() as db:
                path.replace(target)
                file_id = str(uuid4())
                db.execute(
                    "INSERT INTO files VALUES (?,?,?,?,?,?,?)",
                    (
                        file_id,
                        session_id,
                        None,
                        name,
                        str(target.relative_to(store.root)),
                        "upload",
                        size,
                    ),
                )
                db.execute("DELETE FROM upload_reservations WHERE id=?", (reservation,))
                db.execute(
                    "UPDATE sessions SET last_activity=? WHERE id=?",
                    (time.time(), session_id),
                )
        finally:
            path.unlink(missing_ok=True)
            store.execute("DELETE FROM upload_reservations WHERE id=?", (reservation,))
        return file_view(store.one("SELECT * FROM files WHERE id=?", (file_id,)))

    @app.post("/api/v1/sessions/{session_id}/runs", status_code=202)
    def submit(session_id: str, data: RunInput, request: Request):
        session = owned_session(request, session_id)
        if not data.query.strip():
            raise HTTPException(
                422, "Describe what you want to do with the attached files."
            )
        attachments = [str(value) for value in data.attachments]
        for file_id in attachments:
            if not store.one(
                "SELECT id FROM files WHERE id=? AND session_id=? AND kind='upload'",
                (file_id, session_id),
            ):
                raise HTTPException(404, "Attachment not found in this conversation.")
        previous = store.one(
            "SELECT * FROM runs WHERE session_id=? AND request_id=?",
            (session_id, str(data.request_id)),
        )
        if previous:
            if (
                previous["query"] != data.query
                or json.loads(previous["attachments"]) != attachments
            ):
                raise HTTPException(
                    409, "This request ID was already used for a different submission."
                )
            return run_view(previous)
        readiness = session_status(settings, session)
        if not readiness["configured"]:
            raise HTTPException(
                503, {"code": readiness["code"], "message": readiness["message"]}
            )
        run_id = str(uuid4())
        try:
            with store.transaction() as db:
                # Recheck inside the transaction for concurrent retries.
                duplicate = db.execute(
                    "SELECT * FROM runs WHERE session_id=? AND request_id=?",
                    (session_id, str(data.request_id)),
                ).fetchone()
                if duplicate:
                    if (
                        duplicate["query"] != data.query
                        or json.loads(duplicate["attachments"]) != attachments
                    ):
                        raise HTTPException(
                            409,
                            "This request ID was already used for a different submission.",
                        )
                    return run_view(dict(duplicate))
                if db.execute(
                    "SELECT 1 FROM upload_reservations WHERE session_id=?",
                    (session_id,),
                ).fetchone():
                    raise HTTPException(
                        409, "Wait for attachments to finish uploading."
                    )
                if db.execute(
                    "SELECT 1 FROM runs WHERE session_id=? AND status IN ('queued','running','waiting_for_input','cancelling')",
                    (session_id,),
                ).fetchone():
                    raise HTTPException(
                        409,
                        "This conversation already has an unfinished run. Refresh its status.",
                    )
                queued = db.execute(
                    "SELECT count(*) FROM runs WHERE status='queued'"
                ).fetchone()[0]
                owner_runs = db.execute(
                    "SELECT count(*) FROM runs r JOIN sessions s ON s.id=r.session_id WHERE s.owner=? AND r.status IN ('queued','running','waiting_for_input','cancelling')",
                    (request.state.owner,),
                ).fetchone()[0]
                if (
                    queued >= settings.max_queued
                    or owner_runs >= settings.max_user_runs
                ):
                    raise HTTPException(
                        429,
                        {
                            "code": "queue_limit",
                            "message": "The run limit has been reached. Wait for an existing run to finish.",
                        },
                    )
                if (
                    store.owner_usage(request.state.owner, db)
                    + len(data.query.encode())
                    > settings.user_storage_limit
                ):
                    raise HTTPException(
                        413,
                        {
                            "code": "storage_limit",
                            "message": "Your workspace storage limit has been reached.",
                        },
                    )
                db.execute(
                    "INSERT INTO runs(id,session_id,query,request_id,attachments) VALUES (?,?,?,?,?)",
                    (
                        run_id,
                        session_id,
                        data.query,
                        str(data.request_id),
                        json.dumps(attachments),
                    ),
                )
                db.execute(
                    "UPDATE sessions SET last_activity=? WHERE id=?",
                    (time.time(), session_id),
                )
                db.execute(
                    "INSERT INTO events(run_id,type,data) VALUES (?,?,?)",
                    (run_id, "status", json.dumps({"status": "queued"})),
                )
                db.execute(
                    "UPDATE sessions SET title=? WHERE id=? AND title='New conversation'",
                    (data.query[:80], session_id),
                )
        except sqlite3.IntegrityError:
            raise HTTPException(
                409,
                "This conversation already has an unfinished run. Refresh its status.",
            ) from None
        return run_view(store.one("SELECT * FROM runs WHERE id=?", (run_id,)))

    @app.post("/api/v1/runs/{run_id}/cancel", status_code=202)
    def cancel(run_id: str, request: Request):
        owned_run(request, run_id)
        with store.transaction() as db:
            row = db.execute("SELECT status FROM runs WHERE id=?", (run_id,)).fetchone()
            if row["status"] not in TERMINAL and row["status"] != "cancelling":
                state = "cancelled" if row["status"] == "queued" else "cancelling"
                db.execute(
                    "UPDATE runs SET status=?,question=NULL,question_id=NULL,error_code='cancelled',error='The run was cancelled.' WHERE id=?",
                    (state, run_id),
                )
                db.execute(
                    "INSERT INTO events(run_id,type,data) VALUES (?,?,?)",
                    (run_id, "status", json.dumps({"status": state})),
                )
                db.execute(
                    "UPDATE sessions SET last_activity=? WHERE id=(SELECT session_id FROM runs WHERE id=?)",
                    (time.time(), run_id),
                )
        return run_view(store.one("SELECT * FROM runs WHERE id=?", (run_id,)))

    @app.get("/api/v1/runs/{run_id}")
    def run_status(run_id: str, request: Request):
        return run_view(owned_run(request, run_id))

    @app.post("/api/v1/runs/{run_id}/response", status_code=202)
    def respond(run_id: str, data: HumanResponse, request: Request):
        owned_run(request, run_id)
        if not data.answer.strip():
            raise HTTPException(422, "A response must not be blank.")
        with store.transaction() as db:
            row = db.execute(
                "SELECT question FROM runs WHERE id=?", (run_id,)
            ).fetchone()
            changed = db.execute(
                "UPDATE runs SET answer=? WHERE id=? AND question_id=? AND answer IS NULL AND status='waiting_for_input'",
                (data.answer, run_id, str(data.question_id)),
            ).rowcount
            if not changed:
                raise HTTPException(
                    409,
                    "This question has already been answered or is no longer pending.",
                )
            db.execute(
                "INSERT INTO events(run_id,type,data) VALUES (?,?,?)",
                (
                    run_id,
                    "human_response",
                    json.dumps({"question": row["question"], "answer": data.answer}),
                ),
            )
            db.execute(
                "UPDATE sessions SET last_activity=? WHERE id=(SELECT session_id FROM runs WHERE id=?)",
                (time.time(), run_id),
            )
        return {"accepted": True}

    @app.get("/api/v1/runs/{run_id}/events")
    async def events(run_id: str, request: Request, after: int = 0):
        owned_run(request, run_id)
        try:
            cursor = max(after, int(request.headers.get("Last-Event-ID", "0")), 0)
        except ValueError:
            raise HTTPException(422, "Invalid event cursor.") from None

        async def stream():
            nonlocal cursor
            idle = 0
            while not await request.is_disconnected():
                rows = store.rows(
                    "SELECT * FROM events WHERE run_id=? AND id>? ORDER BY id LIMIT 200",
                    (run_id, cursor),
                )
                for row in rows:
                    cursor = row["id"]
                    yield f"id: {cursor}\nevent: {row['type']}\ndata: {row['data']}\n\n"
                row = store.one("SELECT status FROM runs WHERE id=?", (run_id,))
                if row["status"] in TERMINAL and len(rows) < 200:
                    yield "event: end\ndata: {}\n\n"
                    return
                idle += 1
                if idle % 30 == 0:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/artifacts/{file_id}/content")
    def download(file_id: str, request: Request):
        row, path = owned_file(request, file_id)
        return FileResponse(
            path,
            filename=row["name"],
            media_type="application/octet-stream",
            headers={"Content-Security-Policy": "sandbox"},
        )

    @app.get("/api/v1/artifacts/{file_id}/structure")
    def preview(file_id: str, request: Request):
        from chemgraph.api.chemistry import structure_xyz

        _, path = owned_file(request, file_id)
        if path.stat().st_size > settings.upload_limit:
            raise HTTPException(413, "This artifact is too large to preview.")
        try:
            return PlainTextResponse(structure_xyz(path))
        except Exception:
            raise HTTPException(
                422,
                "Unable to preview this structure. Download the original artifact instead.",
            ) from None

    return app
