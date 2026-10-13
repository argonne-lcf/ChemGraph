import pytest
pytest.importorskip("a2a")

import asyncio
import importlib.util
import io
import json
import sys
import urllib.error
from contextlib import asynccontextmanager
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

from tests.service.conftest import TOKENS, ArtifactAdapter, running

from chemgraph.service.adapters import Outcome
from tests.service.fixtures import FixtureAdapter

PLUGIN = Path(__file__).parents[2] / "integrations/hermes/__init__.py"
spec = importlib.util.spec_from_file_location("connect_hermes_test", PLUGIN)
plugin = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plugin)


def test_plugin_submits_nonblocking_standard_a2a(monkeypatch):
    captured = []

    def rpc(method, params):
        captured.append((method, params))
        return {"task": {"id": "task", "contextId": "context"}}

    monkeypatch.setattr(plugin, "rpc", rpc)
    receipt = json.loads(
        plugin.send({"text": "calculate", "context_id": "context", "message_id": "stable"})
    )
    assert receipt["message_id"] == "stable"
    assert receipt["acceptance"] == "accepted"
    assert receipt["task"]["id"] == "task"
    method, params = captured[0]
    assert method == "SendMessage"
    assert params["configuration"] == {"returnImmediately": True}
    assert params["message"]["messageId"] == "stable"


def test_approval_is_never_registered_as_model_tool(monkeypatch):
    tools, commands = [], []
    ctx = SimpleNamespace(
        register_tool=lambda **kwargs: tools.append(kwargs),
        register_command=lambda *args, **kwargs: commands.append((args, kwargs)),
    )
    monkeypatch.setenv("CHEMGRAPH_CONNECT_APPROVERS", '["telegram:123"]')
    plugin.register(ctx)
    assert {t["name"] for t in tools} == {
        "cg_submit",
        "cg_status",
        "cg_tasks",
        "cg_cancel",
        "cg_artifacts",
    }
    assert not any("approve" in t["name"] for t in tools)
    assert commands[0][0] == ("cgapprove",)


@pytest.mark.parametrize(
    "platform,user,chat_type,allowed",
    [
        ("telegram", "123", "dm", True),
        ("a2a", "123", "dm", False),
        ("telegram", "other", "dm", False),
        ("telegram", "123", "group", False),
        ("whatsapp_cloud", "555", "dm", True),
        (None, None, None, False),
    ],
)
def test_linked_operator_context_not_model_text(monkeypatch, platform, user, chat_type, allowed):
    fields = {
        "_SESSION_PLATFORM": platform,
        "_SESSION_USER_ID": user,
        "_SESSION_CHAT_TYPE": chat_type,
    }
    context = SimpleNamespace(
        **{key: ContextVar(key, default=value) for key, value in fields.items()}
    )
    monkeypatch.setitem(sys.modules, "gateway", SimpleNamespace(session_context=context))
    monkeypatch.setenv(
        "CHEMGRAPH_CONNECT_APPROVERS", json.dumps(["telegram:123", "whatsapp_cloud:555"])
    )
    # Fake environment identity must never authorize an unbound gateway context.
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", "telegram")
    monkeypatch.setenv("HERMES_SESSION_USER_ID", "123")
    if allowed:
        assert plugin.trusted_actor() == f"{platform}:{user}"
    else:
        with pytest.raises(PermissionError):
            plugin.trusted_actor()


@asynccontextmanager
async def connected_plugin(settings, monkeypatch, adapter=None):
    """Exercise the plugin's urllib/JSON-RPC wire against the real ASGI service.

    Only the socket boundary is replaced; no Hermes runtime or live service is
    needed, including in the core-only environment.
    """
    async with running(settings, adapter) as (service, client, app):
        loop = asyncio.get_running_loop()

        def urlopen(request, timeout):
            assert request.get_header("A2a-version") == "1.0"
            future = asyncio.run_coroutine_threadsafe(
                client.http.request(
                    request.get_method(),
                    request.full_url,
                    headers=dict(request.header_items()),
                    content=request.data,
                ),
                loop,
            )
            response = future.result(timeout=5)
            if response.is_error:
                raise urllib.error.HTTPError(
                    request.full_url,
                    response.status_code,
                    "Test HTTP error",
                    {},
                    io.BytesIO(response.content),
                )
            return io.BytesIO(response.content)

        async def call(handler, args):
            return json.loads(await asyncio.wait_for(asyncio.to_thread(handler, args), 10))

        monkeypatch.setenv("CHEMGRAPH_CONNECT_URL", "http://test")
        monkeypatch.setenv("CHEMGRAPH_CONNECT_TOKEN", TOKENS["alice"])
        monkeypatch.setattr(plugin.urllib.request, "urlopen", urlopen)
        yield SimpleNamespace(service=service, client=client, app=app, call=call, urlopen=urlopen)


async def test_plugin_roundtrip_restart_and_context_followup(settings, monkeypatch):
    async with connected_plugin(settings, monkeypatch, ArtifactAdapter(settings)) as wire:
        receipt = await wire.call(plugin.send, {"text": "Make a note"})
        task = receipt["task"]
        assert receipt["message_id"]
        await wire.service.wait(task["id"], "alice")
        thread = (await wire.service.store.get(task["id"]))["thread_id"]
    async with connected_plugin(settings, monkeypatch, ArtifactAdapter(settings)) as wire:
        recovered = await wire.call(plugin.tasks, {"context_id": task["contextId"]})
        assert [t["id"] for t in recovered["tasks"]] == [task["id"]]
        report = await wire.call(plugin.status, {"task_id": task["id"]})
        assert report["status"]["state"] == "TASK_STATE_COMPLETED"
        assert "Test artifact ready" in report["status"]["message"]["parts"][0]["text"]
        artifacts = await wire.call(plugin.artifacts, {"task_id": task["id"]})
        assert len(artifacts["artifacts"]) == 1
        artifact_id = artifacts["artifacts"][0]["artifactId"]
        assert (await wire.client.http.get(f"/artifacts/{artifact_id}")).json() == {
            "message": "Make a note"
        }
        followup = await wire.call(
            plugin.send, {"text": "Another note", "context_id": task["contextId"]}
        )
        assert followup["task"]["id"] != task["id"]
        assert (await wire.service.store.get(followup["task"]["id"]))["thread_id"] == thread


async def test_plugin_lost_receipt_replays_identical_message_once(settings, monkeypatch):
    async with connected_plugin(settings, monkeypatch) as wire:
        submissions = []

        def lose_response(request, timeout):
            response = wire.urlopen(request, timeout)
            body = json.loads(request.data)
            if body["method"] == "SendMessage":
                submissions.append(body["params"])
                response.close()
                raise TimeoutError("response lost after durable acceptance")
            return response

        monkeypatch.setattr(plugin.urllib.request, "urlopen", lose_response)
        unknown = await wire.call(plugin.send, {"text": "Do this once"})
        assert unknown["acceptance"] == "unknown"
        assert unknown["message_id"]
        assert len(submissions) == 1  # The plugin never retries on its own.
        saved = await wire.call(plugin.tasks, {})
        assert len(saved["tasks"]) == 1
        identifier = saved["tasks"][0]["id"]
        monkeypatch.setattr(plugin.urllib.request, "urlopen", wire.urlopen)
        replay = await wire.call(
            plugin.send, {"text": "Do this once", "message_id": unknown["message_id"]}
        )
        assert replay["acceptance"] == "accepted"
        assert replay["task"]["id"] == identifier
        conflict = await wire.call(
            plugin.send, {"text": "Changed work", "message_id": unknown["message_id"]}
        )
        assert conflict["acceptance"] == "rejected"
        assert (await wire.call(plugin.tasks, {}))["totalSize"] == 1


class QuestionsAdapter(FixtureAdapter):
    async def run(self, record):
        if record["action"] == "resume":
            return Outcome(text=json.dumps(record["responses"], sort_keys=True))
        return Outcome(
            state="input_required",
            reason="clarification",
            text="Which molecule and charge?",
            interrupts=[
                {"id": "molecule", "payload": "Molecule?"},
                {"id": "charge", "payload": "Charge?"},
            ],
        )


async def test_plugin_multiple_clarifications_and_stale_ids(settings, monkeypatch):
    async with connected_plugin(settings, monkeypatch, QuestionsAdapter(settings)) as wire:
        task = (await wire.call(plugin.send, {"text": "Study a molecule"}))["task"]
        await wire.service.wait(task["id"], "alice")
        pending = await wire.call(plugin.status, {"task_id": task["id"]})
        assert {i["id"] for i in pending["metadata"]["interrupts"]} == {"molecule", "charge"}
        wrong = await wire.call(
            plugin.send,
            {
                "text": "Answers",
                "task_id": task["id"],
                "interrupt_responses": {"molecule": "water", "stale": "neutral"},
            },
        )
        assert wrong["acceptance"] == "rejected"
        answer = await wire.call(
            plugin.send,
            {
                "text": "Answers",
                "task_id": task["id"],
                "interrupt_responses": {"molecule": "water", "charge": "neutral"},
            },
        )
        assert answer["task"]["id"] == task["id"]
        finished = await wire.service.wait(task["id"], "alice")
        assert json.loads(finished.status.message.parts[0].text) == {
            "molecule": "water",
            "charge": "neutral",
        }


async def test_plugin_single_clarification_and_operator_boundary(settings, monkeypatch):
    async with connected_plugin(settings, monkeypatch) as wire:
        task = (await wire.call(plugin.send, {"text": "clarify"}))["task"]
        await wire.service.wait(task["id"], "alice")
        wrong = await wire.call(
            plugin.send, {"text": "water", "task_id": task["id"], "interrupt_id": "stale"}
        )
        assert wrong["acceptance"] == "rejected"
        reply = await wire.call(
            plugin.send, {"text": "water", "task_id": task["id"], "interrupt_id": task["id"]}
        )
        assert reply["acceptance"] == "accepted"
        await wire.service.wait(task["id"], "alice")
        pending = (await wire.call(plugin.send, {"text": "clarify approval"}))["task"]
        await wire.service.wait(pending["id"], "alice")
        record = await wire.service.store.get(pending["id"])
        record["reason"] = "approval"
        async with wire.service.lock:
            async with wire.service.store.transaction():
                await wire.service.store.put(record)
        denied = await wire.call(plugin.send, {"text": "approved", "task_id": pending["id"]})
        assert denied["acceptance"] == "rejected"
        assert "operator" in denied["error"]["message"]


async def test_plugin_pagination_caller_isolation_and_cancel(settings, monkeypatch):
    async with connected_plugin(settings, monkeypatch, ArtifactAdapter(settings)) as wire:
        identifiers = set()
        for n in range(3):
            task = (await wire.call(plugin.send, {"text": str(n), "context_id": "study"}))["task"]
            await wire.service.wait(task["id"], "alice")
            identifiers.add(task["id"])
        page = await wire.call(plugin.tasks, {"context_id": "study", "page_size": 1})
        found = {t["id"] for t in page["tasks"]}
        while page.get("nextPageToken"):
            page = await wire.call(
                plugin.tasks,
                {"context_id": "study", "page_size": 1, "page_token": page["nextPageToken"]},
            )
            found.update(t["id"] for t in page["tasks"])
        assert found == identifiers
        assert not (await wire.call(plugin.tasks, {"context_id": "other"})).get("tasks")
        monkeypatch.setenv("CHEMGRAPH_CONNECT_TOKEN", TOKENS["bob"])
        assert not (await wire.call(plugin.tasks, {"context_id": "study"})).get("tasks")
        for handler in (plugin.status, plugin.cancel, plugin.artifacts):
            with pytest.raises(plugin.RPCError, match="not found"):
                await wire.call(handler, {"task_id": task["id"]})
    async with connected_plugin(settings, monkeypatch) as wire:
        task = (await wire.call(plugin.send, {"text": "clarify cancellation"}))["task"]
        await wire.service.wait(task["id"], "alice")
        canceled = await wire.call(plugin.cancel, {"task_id": task["id"]})
        assert canceled["status"]["state"] == "TASK_STATE_CANCELED"


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"result": {}}',
        b'{"error": {"code": -32603, "message": "Projection failed"}}',
    ],
)
def test_invalid_or_internal_error_receipt_is_uncertain(monkeypatch, body):
    monkeypatch.setenv("CHEMGRAPH_CONNECT_URL", "http://test")
    monkeypatch.setenv("CHEMGRAPH_CONNECT_TOKEN", "test-token")
    monkeypatch.setattr(plugin.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    receipt = json.loads(plugin.send({"text": "work", "message_id": "keep-this-id"}))
    assert receipt["acceptance"] == "unknown"
    assert receipt["message_id"] == "keep-this-id"


@pytest.mark.parametrize(
    "code,acceptance", [(401, "rejected"), (403, "rejected"), (503, "unknown")]
)
def test_submission_http_errors_preserve_id_without_secrets(monkeypatch, code, acceptance):
    def fail(*args):
        raise urllib.error.HTTPError("https://secret.invalid", code, "private-token", {}, None)

    monkeypatch.setattr(plugin, "request", fail)
    receipt = json.loads(plugin.send({"text": "work"}))
    assert receipt["message_id"]
    assert receipt["acceptance"] == acceptance
    assert "private-token" not in json.dumps(receipt)
    assert "secret.invalid" not in json.dumps(receipt)


def test_missing_configuration_is_not_sent(monkeypatch):
    monkeypatch.delenv("CHEMGRAPH_CONNECT_TOKEN", raising=False)
    monkeypatch.setenv("CHEMGRAPH_CONNECT_URL", "http://test")
    receipt = json.loads(plugin.send({"text": "work"}))
    assert receipt["message_id"]
    assert receipt["acceptance"] == "not_sent"


pytestmark = pytest.mark.asyncio
