"""Independent Hermes notification helper; only httpx is required."""

import argparse
import asyncio
import json
import os
from pathlib import Path

async def hermes_send(executable, target, text):
    """No shell, dynamic recipients or MEDIA directives from model output."""
    try:
        process = await asyncio.create_subprocess_exec(
            executable,
            "send",
            "--to",
            target,
            "--json",
            "--file",
            "-",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError:
        return "retry", "Hermes could not be started"
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(text.encode()), timeout=60)
    except TimeoutError:
        process.kill()
        await process.wait()
        return "ambiguous", "Hermes timed out; delivery may have occurred"
    if process.returncode == 0:
        return "delivered", stdout.decode(errors="replace")[:4000]
    # Backend errors can follow acceptance; classify conservatively.
    return "ambiguous", stderr.decode(errors="replace")[:4000]


async def deliver_once(client, recipients, executable="hermes", send=hermes_send):
    response = await client.post("/notifications/claim")
    response.raise_for_status()
    notification = response.json()["notification"]
    if notification is None:
        return False
    target = recipients.get(notification["owner"])
    if not target or not target.startswith("telegram:"):
        state, receipt = "ambiguous", "No fixed Telegram recipient configured for task owner"
    else:
        # Delayed notifications always use linked Telegram, including WhatsApp requests.
        text = (
            f"ChemGraph task {notification['task_id']}: {notification['state']}. "
            f"Notification {notification['id']}. Use task status to retrieve the result."
        )
        state, receipt = await send(executable, target, text)
    response = await client.post(
        "/notifications/ack", json={"id": notification["id"], "state": state, "receipt": receipt}
    )
    response.raise_for_status()
    return True


async def run(args):
    import httpx

    recipients = json.loads(Path(args.recipients).read_text())
    async with httpx.AsyncClient(base_url=args.url, timeout=90, headers={
        "Authorization": f"Bearer {os.environ[args.token_env]}"
    }) as client:
        while True:
            await deliver_once(client, recipients, args.hermes)
            if args.once:
                return
            await asyncio.sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-env", default="CONNECT_NOTIFIER_TOKEN")
    parser.add_argument("--recipients", required=True)
    parser.add_argument("--hermes", default="hermes")
    parser.add_argument("--once", action="store_true")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
