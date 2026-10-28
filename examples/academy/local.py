"""Two persistent ChemGraph sessions using Academy's local exchange."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Keep the sibling http.py from shadowing the standard-library package.
    sys.path[0] = str(Path(__file__).resolve().parents[2])

from academy.exchange import LocalExchangeFactory
from academy.handle import Handle
from academy.manager import Manager

from chemgraph.academy import ChemGraphPeer, PeerMessage, send_peer_message
from examples.academy.common import SessionHandler, console_call


async def main(model, work_dir):
    async with await Manager.from_exchange_factory(LocalExchangeFactory()) as manager:
        registrations = {name: await manager.exchange_client.register_agent(ChemGraphPeer, name=name)
                         for name in ("planner", "executor")}
        handles = {name: Handle(reg.agent_id, exchange=manager.exchange_client)
                   for name, reg in registrations.items()}
        lock = asyncio.Lock()
        handlers = {}
        for name in registrations:
            handler = SessionHandler(name, {other: handle for other, handle in handles.items() if other != name},
                                     model=model, work_dir=work_dir, console_lock=lock)
            handlers[name] = handler
            await manager.launch(ChemGraphPeer(handler, on_error=handler.on_error), registration=registrations[name])

        message = PeerMessage(
            sender="operator",
            content=(
                "Optimize the geometry of H2 using MACE small and EMT, "
                "then compare the final H–H bond lengths in angstroms."
            ),
        )
        print(await send_peer_message(handles["planner"], message), flush=True)
        # The first completed turn delegates; the later turn consumes the
        # executor's reply. Both turns use the same MainAgentSession.
        while (await handlers["planner"].completed.get()).sender != "executor":
            pass
        async with lock:
            await console_call(input, "Reply received; peers are still alive. Enter to shut down: ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.getenv("CHEMGRAPH_MODEL", "gpt-4o-mini"))
    parser.add_argument("--work-dir", default="academy-local")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.model, args.work_dir))
    except KeyboardInterrupt:
        pass
