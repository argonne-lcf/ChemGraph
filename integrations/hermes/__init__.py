"""Hermes plugin; stdlib only, installed in Hermes's own environment."""

import asyncio
import json
import os
import re
import shlex
import urllib.error
import urllib.request
from http.client import HTTPException
from uuid import uuid4


def request(path, data=None, *, operator=False):
    url = os.environ["CHEMGRAPH_CONNECT_URL"].rstrip("/")
    token = os.environ[
        "CHEMGRAPH_CONNECT_OPERATOR_TOKEN" if operator else "CHEMGRAPH_CONNECT_TOKEN"
    ]
    headers = {
        "Authorization": f"Bearer {token}",
        "A2A-Version": "1.0",
        "Content-Type": "application/json",
    }
    req = urllib.request.Request(
        url + path, headers=headers, data=json.dumps(data).encode() if data is not None else None
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())


class RPCError(ValueError):
    def __init__(self, error):
        self.code = error.get("code")
        super().__init__(error.get("message", "Connect returned a JSON-RPC error"))


def rpc(method, params):
    response = request(
        "/", {"jsonrpc": "2.0", "id": str(uuid4()), "method": method, "params": params}
    )
    if not isinstance(response, dict):
        raise ValueError("Invalid JSON-RPC response")
    if isinstance(response.get("error"), dict):
        raise RPCError(response["error"])
    if not isinstance(response.get("result"), dict):
        raise ValueError("Missing JSON-RPC result")
    return response["result"]


def send(args, **kwargs):
    message = {
        "messageId": args.get("message_id") or str(uuid4()),
        "role": "ROLE_USER",
        "parts": [{"text": args["text"]}],
    }
    for key, wire in (("context_id", "contextId"), ("task_id", "taskId")):
        if args.get(key):
            message[wire] = args[key]
    metadata = {
        key: args[key]
        for key in ("interrupt_id", "interrupt_responses")
        if args.get(key) is not None
    }
    if metadata:
        message["metadata"] = metadata
    result = {"message_id": message["messageId"]}
    try:
        response = rpc(
            "SendMessage", {"message": message, "configuration": {"returnImmediately": True}}
        )
        if not isinstance(response.get("task"), dict) or not response["task"].get("id"):
            raise ValueError("Missing task receipt")
        return json.dumps({**response, **result, "acceptance": "accepted"})
    except RPCError as exc:
        # An internal server error can occur AFTER durable acceptance. Only
        # protocol/validation rejections establish that this request was rejected.
        result.update(
            acceptance="rejected" if exc.code in {-32600, -32601, -32602} else "unknown",
            error={"type": "rpc", "code": exc.code, "message": str(exc)},
        )
    except urllib.error.HTTPError as exc:
        result.update(
            acceptance="rejected" if exc.code in {401, 403} else "unknown",
            error={"type": "http", "code": exc.code, "message": "Connect HTTP request failed"},
        )
    except KeyError:
        result.update(
            acceptance="not_sent",
            error={"type": "configuration", "message": "Configure the Connect URL and peer token"},
        )
    except (OSError, HTTPException, ValueError):
        # Do not echo transport exceptions: they can contain URLs or credentials.
        result.update(
            acceptance="unknown",
            error={"type": "transport", "message": "No valid task receipt was received"},
        )
    if result["acceptance"] == "unknown":
        result["recovery"] = (
            "The task may already exist. Inspect cg_tasks/cg_status. If replaying, use this "
            "message_id with exactly the original text, context_id, task_id and interrupt fields. "
            "Never retry with a new ID. Uncertain scientific execution needs operator review."
        )
    return json.dumps(result)


def status(args, **kwargs):
    return json.dumps(rpc("GetTask", {"id": args["task_id"]}))


def tasks(args, **kwargs):
    params = {"pageSize": args.get("page_size", 50)}
    for key, wire in (("context_id", "contextId"), ("page_token", "pageToken")):
        if args.get(key):
            params[wire] = args[key]
    return json.dumps(rpc("ListTasks", params))


def cancel(args, **kwargs):
    return json.dumps(rpc("CancelTask", {"id": args["task_id"]}))


def artifacts(args, **kwargs):
    task = rpc("GetTask", {"id": args["task_id"]})
    return json.dumps(
        {
            "task_id": task["id"],
            "artifacts": task.get("artifacts", []),
            "download": "Use chemgraph-connect artifact with this peer credential. URLs require authentication.",
        }
    )


def trusted_actor():
    # Pinned Hermes ContextVars, deliberately no environment fallback. An A2A
    # caller or model-generated string cannot populate this gateway identity.
    from gateway import session_context

    platform = session_context._SESSION_PLATFORM.get()
    user = session_context._SESSION_USER_ID.get()
    chat_type = session_context._SESSION_CHAT_TYPE.get()
    allowed = json.loads(os.environ.get("CHEMGRAPH_CONNECT_APPROVERS", "[]"))
    actor = f"{platform}:{user}"
    if platform not in {"telegram", "whatsapp_cloud"} or chat_type != "dm" or actor not in allowed:
        raise PermissionError("Approval requires an explicitly linked operator in a direct message")
    return actor


async def approve_command(raw_args):
    try:
        actor = trusted_actor()
        identifier, digest, interrupt_id = shlex.split(raw_args)
        if not re.fullmatch(r"[a-f0-9]{64}", identifier) or not re.fullmatch(
            r"[a-f0-9]{64}", digest
        ):
            raise ValueError("Invalid identifier or digest")
        # This handler is registered ONLY as a slash command, never as a model tool.
        result = await asyncio.to_thread(
            request,
            f"/admin/operations/{identifier}/approve",
            {
                "digest": digest,
                "interrupt_id": interrupt_id,
                "expires_in": 600,
                "chat_actor": actor,
            },
            operator=True,
        )
        return json.dumps(result)
    except (ValueError, PermissionError, KeyError, ImportError) as exc:
        return f"Approval denied: {exc}. CLI approval remains available."


def register(ctx):
    string = {"type": "string"}
    entries = [
        (
            "cg_submit",
            send,
            "Delegate chemistry research and local scientific work to ChemGraph. Returns a task "
            "immediately, not a finished result. Save message_id and task/context IDs; use cg_status "
            "for progress. Reply to clarification using task_id and interrupt fields. For a new "
            "turn after completion use context_id only. After a timeout, inspect cg_tasks or reuse "
            "the exact message and message_id; never submit it with a fresh ID.",
            {
                "text": {"type": "string", "minLength": 1, "maxLength": 100000},
                "context_id": string,
                "task_id": string,
                "message_id": string,
                "interrupt_id": {
                    "type": "string",
                    "description": "The current interrupt ID when answering one clarification.",
                },
                "interrupt_responses": {
                    "type": "object",
                    "description": "Answers keyed by every pending clarification interrupt ID.",
                    "additionalProperties": {"type": "string", "minLength": 1},
                },
            },
            ["text"],
        ),
        (
            "cg_status",
            status,
            "Get the persisted ChemGraph report, progress, or pending questions. INPUT_REQUIRED "
            "with waiting_reason=clarification needs user answers. Approval, configuration and "
            "reconciliation require operator action; ordinary replies cannot authorize them.",
            {"task_id": string},
            ["task_id"],
        ),
        (
            "cg_tasks",
            tasks,
            "Recover ChemGraph task/context IDs across sessions or after an uncertain submission. "
            "Lists only this peer's tasks. Pass nextPageToken as page_token with the same context "
            "filter to fetch more; use cg_status for the selected task's history and report.",
            {
                "context_id": string,
                "page_size": {"type": "integer", "minimum": 1, "maximum": 100, "default": 50},
                "page_token": string,
            },
            [],
        ),
        (
            "cg_cancel",
            cancel,
            "Request cancellation; report whether stopping was confirmed.",
            {"task_id": string},
            ["task_id"],
        ),
        (
            "cg_artifacts",
            artifacts,
            "List authenticated ChemGraph artifacts for a task.",
            {"task_id": string},
            ["task_id"],
        ),
    ]
    for name, handler, description, properties, required in entries:
        ctx.register_tool(
            name=name,
            toolset="chemgraph_connect",
            handler=handler,
            schema={
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        )
    # Disabled until the operator explicitly configures linked identities and a
    # separate control credential. Natural language approval is never accepted.
    if os.environ.get("CHEMGRAPH_CONNECT_APPROVERS"):
        ctx.register_command(
            "cgapprove",
            handler=approve_command,
            description="Approve a reviewed ChemGraph operation",
            args_hint="<operation-id> <digest> <interrupt-id>",
        )
