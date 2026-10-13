"""A2A 1.0 HTTP routes supplied by the official SDK."""

import hmac
from contextlib import asynccontextmanager
from importlib.metadata import version

from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.types import a2a_pb2 as a2a
from a2a.utils.errors import (
    InvalidParamsError,
    PushNotificationNotSupportedError,
    TaskNotFoundError,
    UnsupportedOperationError,
)
from starlette.applications import Starlette
from starlette.authentication import SimpleUser
from starlette.middleware import Middleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .protocol import context, to_dict
from .service import Service


class Authentication:
    def __init__(self, app, credentials):
        self.app, self.credentials = app, credentials

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] in {
            "/health/live",
            "/.well-known/agent-card.json",
        }:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        authorization = headers.get(b"authorization", b"")
        principal = next(
            (
                p
                for token, p in self.credentials.items()
                if hmac.compare_digest(authorization, f"Bearer {token}".encode())
            ),
            None,
        )
        if principal is None:
            return await JSONResponse(
                {"error": "Authentication required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )(scope, receive, send)
        scope["user"] = SimpleUser(principal.name)
        scope["principal"] = principal
        if scope["path"].startswith("/admin/") and principal.role != "operator":
            return await JSONResponse({"error": "Operator required"}, status_code=403)(
                scope, receive, send
            )
        if scope["path"].startswith("/notifications/") and principal.role != "notifier":
            return await JSONResponse({"error": "Notifier required"}, status_code=403)(
                scope, receive, send
            )
        if scope["path"] == "/" and principal.role == "notifier":
            return await JSONResponse({"error": "Peer required"}, status_code=403)(
                scope, receive, send
            )
        await self.app(scope, receive, send)


class Handler(RequestHandler):
    def __init__(self, service):
        self.service = service

    async def on_message_send(self, params, context):
        owner = context.user.user_name
        task = await self.service.accept(params, owner)
        if params.configuration.return_immediately:
            return task
        return await self.service.wait(task.id, owner)

    async def on_get_task(self, params, context):
        if params.HasField("history_length") and params.history_length < 0:
            raise InvalidParamsError("historyLength must be nonnegative")
        task = await self.service.get(params.id, context.user.user_name)
        if params.HasField("history_length"):
            history = list(task.history)[-params.history_length :] if params.history_length else []
            del task.history[:]
            task.history.extend(history)
        return task

    async def on_list_tasks(self, params, context):
        return await self.service.tasks.list(params, context)

    async def on_cancel_task(self, params, context):
        return await self.service.cancel(params.id, context.user.user_name)

    async def on_message_send_stream(self, params, context):
        raise UnsupportedOperationError("Streaming is disabled; use polling")
        yield

    on_subscribe_to_task = on_message_send_stream

    async def on_create_task_push_notification_config(self, params, context):
        raise PushNotificationNotSupportedError()

    on_get_task_push_notification_config = on_create_task_push_notification_config
    on_list_task_push_notification_configs = on_create_task_push_notification_config
    on_delete_task_push_notification_config = on_create_task_push_notification_config

    async def on_get_extended_agent_card(self, params, context):
        raise UnsupportedOperationError()


def agent_card(settings):
    return a2a.AgentCard(
        name="ChemGraph Connect",
        description="Persistent scientific conversations with ChemGraph",
        version=version("chemgraph"),
        supported_interfaces=[
            a2a.AgentInterface(
                url=settings.public_url.rstrip("/") + "/",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            )
        ],
        capabilities=a2a.AgentCapabilities(streaming=False, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        security_schemes={
            "bearer": a2a.SecurityScheme(
                http_auth_security_scheme=a2a.HTTPAuthSecurityScheme(scheme="Bearer")
            )
        },
        security_requirements=[a2a.SecurityRequirement(schemes={"bearer": a2a.StringList()})],
        skills=[
            a2a.AgentSkill(
                id="chemgraph.research",
                name="Chemistry research",
                description="Bounded chemistry work, follow-up questions, and authenticated artifacts",
                tags=["chemistry", "research"],
                examples=["Describe your available chemistry tools"],
            )
        ],
    )


def create_app(settings, adapter=None):
    from .notifications import notification_routes
    from .operations import operation_routes

    service = Service(settings, adapter)

    @asynccontextmanager
    async def lifespan(app):
        await service.start()
        try:
            yield
        finally:
            await service.close()

    async def card(request):
        return JSONResponse(to_dict(agent_card(settings)))

    async def live(request):
        return JSONResponse({"alive": True})

    async def ready(request):
        return JSONResponse({"ready": service.ready}, status_code=200 if service.ready else 503)

    async def metrics(request):
        return JSONResponse(await service.metrics())

    async def artifact(request):
        try:
            body, data = await service.artifact(
                request.path_params["artifact_id"], request.user.display_name
            )
        except TaskNotFoundError:
            return JSONResponse({"error": "Artifact unavailable"}, status_code=404)
        return Response(
            body,
            media_type=data["media_type"],
            headers={
                "Content-Disposition": "attachment",
                "X-Content-Type-Options": "nosniff",
                "Cache-Control": "private, no-store",
                "ETag": data["sha256"],
            },
        )

    async def inspect_task(request):
        record = await service.store.get(request.path_params["task_id"])
        if not record:
            return JSONResponse({"error": "Unknown task"}, status_code=404)
        task = await service.tasks.get(record["id"], context(record["owner"]))
        return JSONResponse(to_dict(task))

    async def finalize_task(request):
        record = await service.store.get(request.path_params["task_id"])
        if not record or record["phase"] != "finalization" or not record.get("pending_outcome"):
            return JSONResponse({"error": "No saved task outcome awaits finalization"}, status_code=409)
        from .adapters import Outcome

        await service.finish(record, Outcome(**record["pending_outcome"]))
        return JSONResponse({"phase": record["phase"]})

    app = Starlette(
        lifespan=lifespan,
        routes=[
            Route("/.well-known/agent-card.json", card),
            Route("/health/live", live),
            Route("/health/ready", ready),
            Route("/admin/metrics", metrics),
            Route("/admin/tasks/{task_id}", inspect_task),
            Route("/admin/tasks/{task_id}/finalize", finalize_task, methods=["POST"]),
            Route("/artifacts/{artifact_id}", artifact),
            *create_jsonrpc_routes(Handler(service), "/"),
            *notification_routes(service),
            *operation_routes(service),
        ],
        middleware=[Middleware(Authentication, credentials=settings.credentials())],
    )
    app.state.service = service
    return app
