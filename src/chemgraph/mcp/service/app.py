"""Fresh SDK server instances mounted behind one authenticated ASGI boundary."""

import asyncio
from contextlib import asynccontextmanager
import inspect
from pathlib import Path
import tempfile
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route

from chemgraph.mcp.service.auth import BearerMiddleware, TokenStore, principal
from chemgraph.mcp.service.dispatcher import Dispatcher
from chemgraph.mcp.service.files import QuotaExceeded
from chemgraph.mcp.service.operations import Service
from chemgraph.mcp.service.worker import chemistry_tools


def create_app(config):
    service = Service(config)
    tokens = TokenStore(config.database)
    dispatcher = Dispatcher(service)
    service.dispatcher = dispatcher
    mcp = FastMCP(
        name="ChemGraph Service", stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(
            enable_dns_rebinding_protection=True, allowed_hosts=[config.allowed_host],
            allowed_origins=list(config.allowed_origins)),
        instructions=(
            "Use workspace-relative file paths or artifact:<id> references, never host paths. "
            "Upload small .xyz/.cif/.json inputs with put_workspace_file. "
            "Select a permitted calculator explicitly; EMT is an integration/demo calculator. "
            "run_ase returns submitted and batch_id. Poll check_job_status: approved means queued, "
            "invoking means dispatched, returned means publishing, finalized means a saved outcome. "
            "Inspect outcome and scientific convergence before claiming success. "
            "uncertain means execution cannot be confirmed: do not resubmit automatically. "
            "Use get_job_results and artifact references to inspect/reuse outputs. "
            "read_artifact handles small text; download_path requires the same bearer token."
        ),
    )

    def register_science(spec):
        async def call(**kwargs):
            return await service.call(principal().user_id, spec.name, kwargs)
        call.__name__, call.__doc__ = spec.name, spec.description
        call.__signature__ = inspect.signature(spec.func, eval_str=True).replace(return_annotation=dict[str, Any])
        mcp.add_tool(call, name=spec.name, description=spec.description)

    catalog = chemistry_tools()
    for name in sorted(service.policy.tools):
        register_science(catalog[name])

    @mcp.tool()
    async def check_job_status(batch_id: str) -> dict:
        """Return this user's operation state and observed outcome, if available."""
        return await service.status(principal().user_id, batch_id)

    @mcp.tool()
    async def get_job_results(batch_id: str, include_partial: bool = False) -> dict:
        """Get a saved single-calculation result and immutable artifact references."""
        return await service.results(principal().user_id, batch_id)

    @mcp.tool()
    async def list_jobs(limit: int = 100, offset: int = 0) -> list[dict]:
        """List only your own operations, with bounded pagination."""
        return await service.list_jobs(principal().user_id, limit, offset)

    @mcp.tool()
    async def cancel_job(batch_id: str) -> dict:
        """Withdraw queued work or stop its process; interrupted outcomes stay uncertain."""
        return await dispatcher.cancel(principal().user_id, batch_id)

    @mcp.tool()
    async def put_workspace_file(path: str, content: str) -> dict:
        """Upload bounded UTF-8 .xyz/.cif/.json text into your private workspace."""
        return await service.put_file(principal().user_id, path, content)

    @mcp.tool()
    async def read_artifact(artifact_id: str) -> dict:
        """Read a small UTF-8 artifact that belongs to you."""
        async with service.inline_slots:
            return await asyncio.to_thread(service.files.read, principal().user_id, artifact_id)

    @mcp.tool()
    async def list_artifacts(limit: int = 100, offset: int = 0) -> list[dict]:
        """List your immutable input/output artifacts and reusable references."""
        return await asyncio.to_thread(service.files.list, principal().user_id, limit=limit, offset=offset)

    async def health(request):
        return JSONResponse({"status": "ok"})

    async def ready(request):
        return JSONResponse({"ready": service.ready}, status_code=200 if service.ready else 503)

    async def upload(request):
        return await service.track(ingest_http(request))

    async def ingest_http(request):
        if not service.ready:
            return JSONResponse({"error": "Service is not ready"}, status_code=503)
        owner = principal().user_id
        path = request.query_params.get("path", "")
        service.files.path(owner, path)
        async with service.inline_slots, service.locks[owner]:
            with tempfile.TemporaryFile(dir=config.data_dir) as spool:
                size, base = 0, await asyncio.to_thread(service.files.usage, owner)
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > config.max_upload_bytes:
                        raise QuotaExceeded("Upload byte limit exceeded.")
                    service.files.check_quota_from(base, size * 2)
                    await asyncio.to_thread(spool.write, chunk)
                spool.seek(0)
                chunks = iter(lambda: spool.read(64 * 1024), b"")
                result = await asyncio.to_thread(service.files.ingest, owner, path, chunks)
        return JSONResponse(result, status_code=201)

    async def download(request):
        async with service.inline_slots:
            path, ref = await asyncio.to_thread(service.files.get, principal().user_id, request.path_params["identifier"])
        return FileResponse(path, filename=Path(ref["name"]).name, media_type=ref["media_type"] or "application/octet-stream")

    async def invalid(request, exc):
        code = 413 if isinstance(exc, QuotaExceeded) else 400
        if isinstance(exc, FileNotFoundError):
            code = 404
        elif isinstance(exc, PermissionError):
            code = 403
        return JSONResponse({"error": str(exc)}, status_code=code)

    child = mcp.streamable_http_app()

    @asynccontextmanager
    async def lifespan(app):
        await dispatcher.start()
        try:
            # Mounted Starlette apps do not enter their own lifespan.
            async with mcp.session_manager.run():
                yield
        finally:
            await dispatcher.stop()

    app = Starlette(
        routes=[Route("/healthz", health), Route("/readyz", ready),
                Route("/artifacts", upload, methods=["POST"]),
                Route("/artifacts/{identifier}", download), Mount("/", app=child)],
        lifespan=lifespan,
        exception_handlers={ValueError: invalid, FileNotFoundError: invalid, PermissionError: invalid},
    )
    app.add_middleware(BearerMiddleware, tokens=tokens, config=config)
    app.state.service, app.state.tokens, app.state.mcp = service, tokens, mcp
    return app
