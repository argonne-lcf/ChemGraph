"""Hermetic HTTP tests for authentication, isolation, and real ASE workers."""

import asyncio
from dataclasses import replace
import json
import os

import pytest

from chemgraph.mcp.service.app import create_app
from chemgraph.mcp.service.auth import principal
from chemgraph.runtime.policy import SCOPED_TOOLS
from tests.mcp_service_helpers import WATER, config, running_app, headers, rpc, tool, completed

pytestmark = pytest.mark.skipif(os.name != "posix", reason="The supervised service requires a POSIX host")


@pytest.mark.asyncio
async def test_outer_auth_protects_artifacts_host_origin_and_factory_lifecycle(tmp_path):
    settings = config(tmp_path)
    for _ in range(2):
        async with running_app(settings) as (app, client):
            token = app.state.tokens.issue("alice")
            assert (await client.get("/readyz")).status_code == 200
            for path in ("/mcp", "/mcp/", "/artifacts", "/artifacts/unknown"):
                assert (await client.post(path, json={})).status_code == 401
            assert (await client.get("/artifacts/unknown")).status_code == 401
            assert (await client.get("/healthz", headers={"Host": "evil.test"})).status_code == 421
            assert (await client.get("/healthz", headers={"Origin": "https://evil.test"})).status_code == 403
            response = await rpc(client, token, "tools/list", {})
            names = {entry["name"] for entry in response["tools"]}
            assert SCOPED_TOOLS <= names and "molecule_name_to_smiles" not in names
            assert len(names) == len(SCOPED_TOOLS) + 7
            with pytest.raises(PermissionError):
                principal()


@pytest.mark.asyncio
async def test_public_proxy_host_and_exclusive_instance(tmp_path):
    settings = replace(config(tmp_path), public_url="https://chemgraph.example", allowed_origins=())
    async with running_app(settings) as (app, client):
        token = app.state.tokens.issue("alice")
        assert (await rpc(client, token, "tools/list", {}))["tools"]
        other = create_app(settings)
        with pytest.raises(BlockingIOError):
            async with other.router.lifespan_context(other):
                pass


@pytest.mark.asyncio
async def test_scoped_tools_and_http_artifacts_do_not_cross_users(tmp_path):
    async with running_app(config(tmp_path)) as (app, client):
        alice, bob = [app.state.tokens.issue(name) for name in ("alice", "bob")]
        uploaded = await tool(client, alice, "put_workspace_file", path="water.xyz", content=WATER)
        await tool(client, bob, "put_workspace_file", path="water.xyz", content=WATER.replace("water", "bob"))
        result = await tool(client, alice, "file_to_atomsdata", fname=uploaded["reference"])
        assert result["status"] == "finalized" and result["result"]["numbers"] == [8, 1, 1]
        own = await tool(client, alice, "read_artifact", artifact_id=uploaded["artifact_id"])
        assert own["content"] == WATER
        for identifier in (uploaded["artifact_id"], "unknown"):
            response = await client.get(f"/artifacts/{identifier}", headers=headers(bob))
            assert response.status_code == 404 and response.json() == {"error": "Unknown artifact."}
        for name in ("check_job_status", "get_job_results", "cancel_job"):
            foreign = await tool(client, bob, name, batch_id=result["batch_id"])
            unknown = await tool(client, bob, name, batch_id="unknown")
            assert foreign == unknown == {"error": "Unknown job."}
        assert await tool(client, bob, "list_jobs") == []
        binary = b"\xff\x00\x01"
        response = await client.post("/artifacts?path=blob.bin", headers=headers(alice), content=binary)
        assert response.status_code == 201, response.text
        assert (await client.get(response.json()["download_path"], headers=headers(alice))).content == binary


@pytest.mark.asyncio
async def test_traversal_symlink_reserved_paths_and_streaming_limits(tmp_path):
    settings = config(tmp_path, max_text_bytes=128, max_upload_bytes=256, max_storage_bytes=2048)
    async with running_app(settings) as (app, client):
        token = app.state.tokens.issue("alice")
        owner = app.state.tokens.verify(token).user_id
        root = app.state.service.files.workspace(owner)
        root.mkdir(parents=True)
        (root / "escape").symlink_to(tmp_path, target_is_directory=True)
        for path in ("../x.xyz", "/tmp/x.xyz", "escape/x.xyz", ".chemgraph-inputs/x.xyz", ".service-request.json"):
            response = await rpc(client, token, "tools/call", {
                "name": "put_workspace_file", "arguments": {"path": path, "content": WATER}})
            assert response["isError"]
        async def chunks():
            for _ in range(3):
                yield b"x" * 100
        response = await client.post("/artifacts?path=too-large.xyz", headers=headers(token), content=chunks())
        assert response.status_code == 413
        assert not (root / "too-large.xyz").exists()
        response = await rpc(client, token, "tools/call", {
            "name": "put_workspace_file", "arguments": {"path": "x.xyz", "content": "x" * 129}})
        assert response["isError"]


@pytest.mark.asyncio
async def test_real_emt_jobs_survive_disconnect_and_results_survive_restart(tmp_path):
    settings = config(tmp_path)
    identifiers = []
    async with running_app(settings) as (app, client):
        tokens = [app.state.tokens.issue(name) for name in ("alice", "bob")]
        for token in tokens:
            await tool(client, token, "put_workspace_file", path="water.xyz", content=WATER)
        params = {"input_structure_file": "water.xyz", "output_results_file": "energy.json",
                  "driver": "energy", "calculator": {"calculator_type": "emt"}}
        responses = await asyncio.gather(*(tool(client, token, "run_ase", params=params) for token in tokens))
        identifiers = [result["batch_id"] for result in responses]
        assert len(set(identifiers)) == 2
        assert all(result["status"] == "submitted" for result in responses)
        states = await asyncio.gather(*(completed(client, token, identifier) for token, identifier in zip(tokens, identifiers)))
        assert all(state["status"] == "finalized" and state["outcome"] == "success" for state in states), states
        for token, identifier in zip(tokens, identifiers):
            result = await tool(client, token, "get_job_results", batch_id=identifier)
            assert result["artifacts"], result
            assert str(tmp_path) not in json.dumps(result)
            artifact = next(a for a in result["artifacts"] if a["name"] == "energy.json")
            response = await client.get(artifact["download_path"], headers=headers(token))
            assert response.status_code == 200 and response.json()["potential_energy"] is not None
            assert str(tmp_path) not in response.text
            other = tokens[1] if token == tokens[0] else tokens[0]
            assert (await client.get(artifact["download_path"], headers=headers(other))).status_code == 404
    async with running_app(settings) as (app, client):
        for token, identifier in zip(tokens, identifiers):
            result = await tool(client, token, "get_job_results", batch_id=identifier)
            assert result["status"] == "finalized" and result["artifacts"]
