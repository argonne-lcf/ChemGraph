"""Service and operator command line; no implicit user configuration discovery."""

import argparse
import asyncio
import json
import os
from pathlib import Path


from .client import Client
from .config import load_settings
from .protocol import to_dict


def parser():
    result = argparse.ArgumentParser(prog="chemgraph service")
    result.add_argument("--config", default="config.toml")
    sub = result.add_subparsers(dest="command", required=True)
    sub.add_parser("serve")
    sub.add_parser("doctor")
    for name in (
        "send",
        "reply",
        "status",
        "cancel",
        "tasks",
        "artifact",
        "approve",
        "reject",
        "operations",
        "reconcile",
        "finalize-task",
    ):
        command = sub.add_parser(name)
        command.add_argument("--url", default="http://127.0.0.1:8080")
        command.add_argument("--token-env", default="CONNECT_TOKEN")
        if name in {"status", "cancel", "reply", "operations", "finalize-task"}:
            command.add_argument("task_id")
        if name in {"send", "reply"}:
            command.add_argument("text")
            command.add_argument("--context-id", default="")
            command.add_argument("--message-id")
        if name == "reply":
            command.add_argument("--interrupt-id")
        if name == "artifact":
            command.add_argument("artifact_id")
            command.add_argument("output", type=Path)
        if name in {"approve", "reject"}:
            command.add_argument("operation_id")
            command.add_argument("--digest", required=True)
            command.add_argument("--interrupt-id", required=True)
            command.add_argument("--expires-in", type=int, default=600)
            command.add_argument("--message", default="Operator rejected action")
        if name == "reconcile":
            command.add_argument("operation_id")
            command.add_argument("--batch-id")
            command.add_argument("--mode", choices=["job", "result", "nonexecution", "abandon", "finalize"], default="job")
            command.add_argument("--result-file", type=Path)
            command.add_argument("--evidence", required=True)
    return result


async def remote(args):
    token = os.environ[args.token_env]
    async with Client(args.url, token) as client:
        if args.command in {"send", "reply"}:
            value = await client.send(
                args.text,
                task_id=getattr(args, "task_id", ""),
                context_id=args.context_id,
                message_id=args.message_id,
                interrupt_id=getattr(args, "interrupt_id", None),
            )
        elif args.command == "status":
            value = await client.get(args.task_id)
        elif args.command == "cancel":
            value = await client.cancel(args.task_id)
        elif args.command == "tasks":
            value = await client.list()
        elif args.command in {"approve", "reject", "operations", "reconcile", "finalize-task"}:
            if args.command in {"approve", "reject"}:
                response = await client.http.post(
                    f"/admin/operations/{args.operation_id}/{args.command}",
                    json={
                        "digest": args.digest,
                        "interrupt_id": args.interrupt_id,
                        "expires_in": args.expires_in,
                        "message": args.message,
                    },
                )
            elif args.command == "reconcile":
                response = await client.http.post(
                    f"/admin/operations/{args.operation_id}/reconcile",
                    json={"batch_id": args.batch_id, "evidence": args.evidence, "mode": args.mode,
                          "result": json.loads(args.result_file.read_text()) if args.result_file else None},
                )
            elif args.command == "finalize-task":
                response = await client.http.post(f"/admin/tasks/{args.task_id}/finalize")
            else:
                response = await client.http.get(f"/admin/tasks/{args.task_id}/operations")
            response.raise_for_status()
            print(json.dumps(response.json(), indent=2))
            return
        else:
            response = await client.http.get(f"/artifacts/{args.artifact_id}")
            response.raise_for_status()
            with args.output.open("xb") as output:
                output.write(response.content)
            print(args.output)
            return
        print(json.dumps(to_dict(value), indent=2))


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in {"serve", "doctor"}:
        settings = load_settings(args.config)
        settings.credentials()
        if args.command == "doctor":
            from .diagnostics import diagnose

            result = diagnose(settings)
            print(json.dumps(result, indent=2))
            if not result["configured"]:
                raise SystemExit(1)
        else:
            import uvicorn

            from .server import create_app

            uvicorn.run(create_app(settings), host=settings.host, port=settings.port, ws="none")
    else:
        asyncio.run(remote(args))
