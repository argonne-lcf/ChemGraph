"""Exercise a persistent MCP deployment with two users and real EMT jobs."""

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit
import uuid

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class SmokeFailure(RuntimeError):
    """A deployment check failed without submitting replacement work."""


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


def origin(value):
    parts = urlsplit(value)
    require(
        parts.scheme in {"http", "https"} and parts.hostname
        and parts.username is None and parts.password is None
        and not parts.query and not parts.fragment and parts.path in {"", "/"}
        and not any(char.isspace() for char in value),
        "--url must be an origin without credentials, path, query or fragment.",
    )
    require(parts.port is None or 0 < parts.port < 65536, "Invalid URL port.")
    require(parts.scheme == "https" or parts.hostname in {"localhost", "127.0.0.1", "::1"},
            "HTTPS is required except for loopback development.")
    return f"{parts.scheme}://{parts.netloc.lower()}"


def identifier(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def artifact_path(artifact):
    require(identifier(artifact.get("artifact_id")), "Invalid artifact ID.")
    path = f"/artifacts/{artifact['artifact_id']}"
    require(artifact.get("download_path") == path, "Artifact download path must stay on this service.")
    require(identifier(artifact.get("sha256")), "Invalid artifact checksum.")
    require(type(artifact.get("size_bytes")) is int and artifact["size_bytes"] >= 0,
            "Invalid artifact size.")
    return path


def check_bytes(content, artifact):
    require(len(content) == artifact["size_bytes"]
            and hashlib.sha256(content).hexdigest() == artifact["sha256"],
            "Downloaded artifact size/checksum does not match its descriptor.")


class Peer:
    def __init__(self, session, http, url):
        self.session, self.http, self.url = session, http, url

    async def tool(self, name, **arguments):
        result = await self.session.call_tool(name, arguments)
        require(not result.isError, f"{name} returned a tool error; inspect the service logs.")
        content = result.structuredContent
        if content is None:
            content = json.loads(result.content[0].text)
        return content["result"] if isinstance(content, dict) and set(content) == {"result"} else content

    async def download(self, artifact):
        response = await self.http.get(self.url + artifact_path(artifact))
        require(response.status_code == 200, f"Artifact download failed: HTTP {response.status_code}.")
        check_bytes(response.content, artifact)
        return response.content


@asynccontextmanager
async def peers(url, tokens, *, transport=None):
    async with AsyncExitStack() as stack:
        result = []
        for token in tokens:
            http = await stack.enter_async_context(httpx.AsyncClient(
                headers={"Authorization": f"Bearer {token}"}, timeout=30,
                follow_redirects=False, transport=transport,
            ))
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(url + "/mcp", http_client=http))
            session = await stack.enter_async_context(ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=30)))
            await session.initialize()
            names = {tool.name for tool in (await session.list_tools()).tools}
            require({"run_ase", "put_workspace_file", "get_job_results", "read_artifact"} <= names,
                    "The endpoint does not expose the persistent chemistry service tools.")
            result.append(Peer(session, http, url))
        yield result


async def boundary_checks(url, *, local=False, transport=None):
    async with httpx.AsyncClient(timeout=30, follow_redirects=False, transport=transport) as http:
        ready = await http.get(url + "/readyz")
        require(ready.status_code == 200 and ready.json().get("ready") is True, "Service is not ready.")
        for path in ("/mcp", "/artifacts/unknown"):
            require((await http.get(url + path)).status_code == 401, "Unauthenticated access was not rejected.")
        require((await http.get(url + "/healthz", headers={"Origin": "https://smoke.invalid"})).status_code == 403,
                "Disallowed Origin was not rejected.")
        if local:
            require((await http.get(url + "/healthz", headers={"Host": "smoke.invalid"})).status_code == 421,
                    "Disallowed Host was not rejected by local Uvicorn.")
        else:
            print("Host probe: not tested (a reverse proxy may intercept it).")


async def foreign_artifact(peer, artifact):
    unknown = uuid.uuid4().hex * 2
    foreign = await peer.http.get(peer.url + artifact_path(artifact))
    absent = await peer.http.get(peer.url + f"/artifacts/{unknown}")
    message = "Isolation failed: tokens may identify the same user, or deployment isolation is broken."
    require(foreign.status_code == absent.status_code == 404 and foreign.json() == absent.json(), message)
    foreign_read = await peer.session.call_tool("read_artifact", {"artifact_id": artifact["artifact_id"]})
    absent_read = await peer.session.call_tool("read_artifact", {"artifact_id": unknown})
    require(foreign_read.isError and absent_read.isError and foreign_read.content == absent_read.content, message)


async def foreign_job(peer, batch_id):
    for name in ("check_job_status", "get_job_results"):
        foreign = await peer.tool(name, batch_id=batch_id)
        absent = await peer.tool(name, batch_id=uuid.uuid4().hex * 2)
        require(foreign == absent == {"error": "Unknown job."}, "Foreign and unknown job responses differ.")


async def completed(peer, batch_id, timeout):
    async with asyncio.timeout(timeout):
        while True:
            result = await peer.tool("get_job_results", batch_id=batch_id)
            state = result.get("status")
            if state not in {"approved", "invoking", "returned"}:
                require(state == "finalized" and result.get("outcome") == "success",
                        f"Job {batch_id} did not succeed (state={state}); no resubmission attempted.")
                return result
            await asyncio.sleep(0.2)


async def inspect_result(peer, result, user):
    energy = result["result"].get("potential_energy")
    require(type(energy) in (float, int) and math.isfinite(energy)
            and result["result"].get("energy_unit") == "eV", "Expected a finite EMT energy in eV.")
    artifacts = result["artifacts"]
    require(all(a.get("operation_id") == user["batch_id"] for a in artifacts), "Artifact belongs to another job.")
    contents = {a["name"]: await peer.download(a) for a in artifacts}
    require("energy.json" in contents and "inputs/water.xyz" in contents, "Missing result or input artifact.")
    require(hashlib.sha256(contents["inputs/water.xyz"]).hexdigest() == user["input_sha256"],
            "Job input differs from this user's uploaded structure.")
    document = json.loads(contents["energy.json"])
    require(document.get("success") is True and document.get("potential_energy") == energy,
            "Result artifact disagrees with the successful job result.")
    return {key: result[key] for key in ("batch_id", "updated_at", "result", "artifacts")}


def new_receipt(url):
    return {"version": 1, "status": "incomplete", "origin": url,
            "prefix": "smoke-" + uuid.uuid4().hex,
            "users": [{"label": label} for label in ("alice", "bob")]}


async def exercise(url, tokens, receipt, timeout, save=lambda: None, *, local=False, transport=None):
    print(f"Run prefix: {receipt['prefix']}")
    await boundary_checks(url, local=local, transport=transport)
    async with peers(url, tokens, transport=transport) as clients:
        for index, (peer, user) in enumerate(zip(clients, receipt["users"])):
            water = f"3\n{user['label']} water\nO 0 0 0\nH 0 0 {0.96 + index * .1}\nH 0.93 0 -0.24\n"
            path = receipt["prefix"] + "/water.xyz"
            if index == 0:
                upload = await peer.tool("put_workspace_file", path=path, content=water)
            else:
                response = await peer.http.post(url + "/artifacts", params={"path": path}, content=water.encode())
                require(response.status_code == 201, f"Upload failed: HTTP {response.status_code}.")
                upload = response.json()
            artifact_path(upload)
            check_bytes(water.encode(), upload)
            user.update(upload=upload, input_sha256=upload["sha256"])
            print(f"{user['label']} upload: {upload['artifact_id']}")
            save()
            # Check the two credentials before any calculation is submitted.
            await foreign_artifact(clients[1 - index], upload)
        for peer, user in zip(clients, receipt["users"]):
            job = await peer.tool("run_ase", params={
                "input_structure_file": user["upload"]["reference"], "output_results_file": "energy.json",
                "driver": "energy", "steps": 1, "calculator": {"calculator_type": "emt"},
            })
            require(identifier(job.get("batch_id")), "Service did not return a valid job ID.")
            user["batch_id"] = job["batch_id"]
            print(f"{user['label']} job: {user['batch_id']}")
            save()
            require(job.get("status") == "submitted", "Submission was blocked; EMT and steps=1 must be permitted.")
    # New HTTP and MCP sessions prove that results do not depend on the submitter's connection.
    async with peers(url, tokens, transport=transport) as clients:
        for index, (peer, user) in enumerate(zip(clients, receipt["users"])):
            result = await completed(peer, user["batch_id"], timeout)
            user.update(await inspect_result(peer, result, user))
            save()
            await foreign_job(clients[1 - index], user["batch_id"])
            for artifact in user["artifacts"]:
                await foreign_artifact(clients[1 - index], artifact)
    receipt["status"] = "complete"
    save()
    print("PASS: two EMT jobs, reconnect, owner isolation, input bytes and artifact checksums.")


def validate_receipt(receipt, url):
    require(receipt.get("version") == 1 and receipt.get("status") == "complete",
            "Receipt is unsupported or incomplete; verify never submits replacement work.")
    require(receipt.get("origin") == url, "Receipt origin does not match --url.")
    users = receipt.get("users")
    require(isinstance(users, list) and len(users) == 2, "Receipt must describe two users.")
    for user, label in zip(users, ("alice", "bob")):
        require(user.get("label") == label and identifier(user.get("batch_id"))
                and identifier(user.get("input_sha256")), "Invalid receipt user/job/input identity.")
        require(isinstance(user.get("result"), dict) and isinstance(user.get("artifacts"), list)
                and user["artifacts"] and type(user.get("updated_at")) in (int, float)
                and math.isfinite(user["updated_at"]), "Invalid saved result in receipt.")
        artifact_path(user["upload"])
        for artifact in user["artifacts"]:
            artifact_path(artifact)


async def verify(url, tokens, receipt, *, local=False, transport=None):
    validate_receipt(receipt, url)  # Before sending credentials or reading any server state.
    await boundary_checks(url, local=local, transport=transport)
    async with peers(url, tokens, transport=transport) as clients:
        for index, (peer, user) in enumerate(zip(clients, receipt["users"])):
            result = await peer.tool("get_job_results", batch_id=user["batch_id"])
            require(result.get("status") == "finalized" and result.get("outcome") == "success",
                    "Saved job is no longer a finalized success.")
            observed = await inspect_result(peer, result, user)
            require(observed == {key: user[key] for key in observed}, "Saved result, timestamp or artifacts changed.")
            await peer.download(user["upload"])
            await foreign_job(clients[1 - index], user["batch_id"])
            for artifact in user["artifacts"]:
                await foreign_artifact(clients[1 - index], artifact)
    print("PASS: saved results, timestamps and artifact checksums unchanged; no jobs submitted.")


@asynccontextmanager
async def local_server(command, root, url, timeout, unsafe):
    log_path = root / "server.log"
    with log_path.open("ab") as log:
        process = await asyncio.create_subprocess_exec(*command, "serve", stdout=log, stderr=log,
                                                       start_new_session=True)
        # Only a confirmed graceful exit makes removing this directory safe.
        unsafe[0] = True
        try:
            async with httpx.AsyncClient(timeout=2, follow_redirects=False) as http:
                async with asyncio.timeout(timeout):
                    while True:
                        require(process.returncode is None, "Local service exited during startup.")
                        try:
                            if (await http.get(url + "/readyz")).status_code == 200:
                                break
                        except httpx.TransportError:
                            pass
                        await asyncio.sleep(.1)
            yield
        finally:
            crashed = process.returncode is not None
            forced = False
            try:
                if not crashed:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=15)
                    except TimeoutError:
                        forced = True
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        await asyncio.wait_for(process.wait(), timeout=5)
                # Uvicorn re-raises our SIGTERM after graceful lifespan shutdown.
                require(not crashed and not forced and process.returncode in (0, -signal.SIGTERM),
                        "Unclean shutdown.")
            except BaseException as error:
                raise SmokeFailure(f"Server did not shut down cleanly. Data/logs retained at {root}; "
                                   "worker groups may still be running. Stop them before cleanup or reuse.") from error
            unsafe[0] = False


async def local(timeout, secrets):
    require(os.name == "posix", "Local service deployment requires a POSIX host.")
    root = Path(tempfile.mkdtemp(prefix="chemgraph-mcp-smoke-"))
    unsafe = [False]
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        settings = root / "service.toml"
        settings.write_text(
            f'[service]\ndata_dir = {json.dumps(str(root / "data"))}\npublic_url = "{url}"\n'
            f'port = {port}\nmax_workers = 2\nshutdown_seconds = 2\nmax_job_seconds = {math.ceil(timeout)}\n'
            'enable_pubchem = false\n[policy]\ncalculators = ["emt"]\n'
            '[policy.limits]\nmax_atoms = 10\nmax_steps = 10\n', encoding="utf-8")
        command = [sys.executable, "-m", "chemgraph.mcp.service", "--config", str(settings)]
        tokens = [subprocess.check_output(command + ["token", "issue", label], text=True, timeout=30).strip()
                  for label in ("smoke-alice", "smoke-bob")]
        secrets.extend(tokens)
        receipt = new_receipt(url)
        async with local_server(command, root, url, min(timeout, 30), unsafe):
            await exercise(url, tokens, receipt, timeout, local=True)
        print("Restarting the local service with the same data.")
        async with local_server(command, root, url, min(timeout, 30), unsafe):
            await verify(url, tokens, receipt, local=True)
    except BaseException:
        log = root / "server.log"
        if log.exists():
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 4000))
                tail = stream.read().decode(errors="replace")
            for secret in secrets:
                tail = tail.replace(secret, "[redacted]")
            print("Local server log (tail):\n" + tail, file=sys.stderr)
        raise
    finally:
        if not unsafe[0]:
            shutil.rmtree(root)


async def remote(args, secrets):
    url = origin(args.url)
    tokens = [os.environ.get(name, "") for name in (args.alice_token_env, args.bob_token_env)]
    require(all(tokens), "Both token environment variables must be nonempty.")
    secrets.extend(tokens)
    if args.mode == "verify":
        await verify(url, tokens, json.loads(args.receipt.read_text(encoding="utf-8")))
        return
    receipt = new_receipt(url)
    # Reserve the destination before ANY request, including probes and discovery.
    fd = os.open(args.receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        def save():
            content = json.dumps(receipt, indent=2, allow_nan=False)
            require(not any(token in content for token in tokens), "Server response contains credentials; receipt withheld.")
            stream.seek(0)
            stream.write(content + "\n")
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        save()
        print(f"Remote data persists; there is no deletion API. Accounting receipt: {args.receipt}")
        await exercise(url, tokens, receipt, args.timeout, save)


def positive_seconds(value):
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("timeout must be a positive finite number")
    return seconds


def describe(error):
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(describe(child) for child in error.exceptions)
    return str(error) or type(error).__name__


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for name in ("local", "remote", "verify"):
        mode = modes.add_parser(name)
        mode.add_argument("--timeout", type=positive_seconds, default=120,
                          help="Startup/job deadline in seconds (default: 120; local startup capped at 30).")
        if name != "local":
            mode.add_argument("--url", required=True, help="Service origin, without /mcp.")
            mode.add_argument("--alice-token-env", required=True)
            mode.add_argument("--bob-token-env", required=True)
            mode.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    secrets = []
    logging.basicConfig(level=logging.CRITICAL)
    try:
        asyncio.run(local(args.timeout, secrets) if args.mode == "local" else remote(args, secrets))
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        message = describe(error)
        for secret in secrets:
            message = message.replace(secret, "[redacted]")
        print("FAIL: " + message, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
