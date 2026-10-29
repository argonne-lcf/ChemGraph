"""Run the same handlers in separate processes with native Academy registrations."""

import argparse
import asyncio
import os
import shlex
import sys
from contextlib import AsyncExitStack
from pathlib import Path

if __package__ in (None, ""):
    # Make the repository root importable for examples.academy.common.
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from academy.exchange import HttpExchangeFactory
from academy.exchange.cloud.client import HttpAgentRegistration
from academy.handle import Handle
from academy.runtime import Runtime, RuntimeConfig
from academy.serialize import SerializationStrategy

from chemgraph.academy import ChemGraphPeer, PeerMessage, send_peer_message
from examples.academy.common import SessionHandler


async def main(args):
    factory = HttpExchangeFactory(args.url, auth_method=None)
    async with AsyncExitStack() as stack:
        client = await factory.create_user_client()
        stack.push_async_callback(client.close)
        if args.command == "register":
            for name in ("planner", "executor"):
                registration = await client.register_agent(ChemGraphPeer, name=name)
                print(f"export {name.upper()}_REGISTRATION={shlex.quote(registration.model_dump_json())}")
            return

        registrations = {name: HttpAgentRegistration.model_validate_json(os.environ[f"{name.upper()}_REGISTRATION"])
                         for name in ("planner", "executor")}
        handles = {name: Handle(reg.agent_id, exchange=client,
                                request_serializer=SerializationStrategy.JSON,
                                result_serializer=SerializationStrategy.JSON,
                                exception_serializer=SerializationStrategy.JSON)
                   for name, reg in registrations.items()}
        if args.command == "send":
            message = PeerMessage(sender="operator", content=args.content)
            async with asyncio.timeout(30):
                print(await send_peer_message(handles["planner"], message))
            return

        handler = SessionHandler(args.name, {name: handle for name, handle in handles.items() if name != args.name},
                                 model=args.model, work_dir=args.work_dir, console_lock=asyncio.Lock())
        async with Runtime(
            ChemGraphPeer(handler, on_error=handler.on_error), exchange_factory=factory,
            registration=registrations[args.name],
            config=RuntimeConfig(default_serializer=SerializationStrategy.JSON,
                                 allowed_deserializers={SerializationStrategy.JSON}),
        ) as runtime:
            print(f"{args.name} ready; Ctrl-C to shut down", flush=True)
            await runtime.wait_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8700")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("register")
    run = commands.add_parser("run")
    run.add_argument("name", choices=("planner", "executor"))
    run.add_argument("--model", default=os.getenv("CHEMGRAPH_MODEL", "gpt-4o-mini"))
    run.add_argument("--work-dir", default="academy-http")
    send = commands.add_parser("send")
    send.add_argument("content")
    try:
        asyncio.run(main(parser.parse_args()))
    except KeyboardInterrupt:
        pass
