"""Production adapters over mocked HTTP/SSE, with no endpoint credentials."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import anthropic
import httpx
import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from chemgraph.agent.usage import UsageCollector
from chemgraph.models.protocols import anthropic_native, openai_compatible
from chemgraph.models.protocols.usage_capture import _active


def body(provider, usage):
    if provider == "anthropic":
        return {"id": "msg_test", "type": "message", "role": "assistant",
                "model": "claude-test", "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn", "stop_sequence": None, "usage": usage}
    return {"id": "chat_test", "object": "chat.completion", "created": 1,
            "model": "gpt-test", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": "hello"},
            "finish_reason": "stop"}], "usage": usage}


def events(provider, inputs=10, outputs=7):
    if provider == "anthropic":
        message = body(provider, {"input_tokens": inputs, "output_tokens": 0,
                                  "future_detail": {"retained": 1}})
        message.update(content=[], stop_reason=None)
        return [
            {"type": "message_start", "message": message},
            {"type": "content_block_start", "index": 0,
             "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0,
             "delta": {"type": "text_delta", "text": "hello"}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
             "usage": {"output_tokens": outputs}},
            {"type": "message_stop"},
        ]
    base = {"id": "chat_test", "object": "chat.completion.chunk", "created": 1, "model": "gpt-test"}
    return [
        {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"},
                              "finish_reason": None}]},
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**base, "choices": [], "usage": {"prompt_tokens": inputs, "completion_tokens": outputs,
                                           "future_detail": {"retained": 1}}},
    ]


class WireStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, content, interrupted):
        self.content, self.interrupted = content, interrupted

    def __iter__(self):
        yield self.content
        if self.interrupted:
            raise httpx.ReadError("disconnected")

    async def __aiter__(self):
        for part in self:
            await asyncio.sleep(0)
            yield part


def client(provider, *, usage=None, stream_events=None, interrupted=False, upstream=False):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        if stream_events is not None:
            content = b"".join(
                (f"event: {e['type']}\n" if provider == "anthropic" else "").encode()
                + f"data: {json.dumps(e)}\n\n".encode() for e in stream_events
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  stream=WireStream(content, interrupted))
        return httpx.Response(200, json=body(provider, usage))

    transport = httpx.MockTransport(handle)
    sync = httpx.Client(transport=transport)
    asynchronous = httpx.AsyncClient(transport=transport)
    kwargs = dict(api_key="test-key", max_retries=0)
    if provider == "openai":
        kwargs.update(model="gpt-test", http_client=sync, http_async_client=asynchronous,
                      stream_usage=True, use_responses_api=False)
        model = ChatOpenAI(**kwargs) if upstream else openai_compatible.build(kwargs)
    else:
        kwargs.update(model="claude-test", max_tokens=30)
        model = ChatAnthropic(**kwargs) if upstream else anthropic_native.build(kwargs)
        model.__dict__["_client"] = anthropic.Anthropic(api_key="test-key", http_client=sync, max_retries=0)
        model.__dict__["_async_client"] = anthropic.AsyncAnthropic(api_key="test-key", http_client=asynchronous, max_retries=0)
    return model, requests


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_stream_captures_before_normalization(provider, asynchronous):
    model, _ = client(provider, stream_events=events(provider))
    collector = UsageCollector("s", "t", capture_evidence=True)
    config = {"callbacks": [collector]}
    if asynchronous:
        chunks = [c async for c in model.astream("hello", config)]
    else:
        chunks = list(model.stream("hello", config))
    record, = collector.records
    assert collector.summary["total_tokens"] == 17
    assert collector.summary["partial"] is False
    assert record["source"] == "provider_raw"
    assert record["raw_capture"] == "captured"
    assert record["raw_usage"]["future_detail"] == {"retained": 1}
    assert record["model"] == ("claude-test" if provider == "anthropic" else "gpt-test")
    assert all("usage_evidence" not in c.model_dump_json() for c in chunks)
    assert _active.get() is None
    if provider == "anthropic":
        # Regression: pinned upstream produces only 7; accounting must retain 10 input.
        assert sum((c.usage_metadata or {}).get("total_tokens", 0) for c in chunks) == 7


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("capture_evidence", [False, True])
async def test_regular_response_and_prompts_unchanged(provider, asynchronous, capture_evidence):
    usage = ({"input_tokens": 10, "output_tokens": 7, "cache_read_input_tokens": 5,
              "cache_creation_input_tokens": 3} if provider == "anthropic" else
             {"prompt_tokens": 18, "completion_tokens": 7})
    usage["future_detail"] = {"retained": 1}
    model, requests = client(provider, usage=usage)
    baseline, baseline_requests = client(provider, usage=usage, upstream=True)
    collector = UsageCollector("s", "t", capture_evidence=capture_evidence)
    messages = [HumanMessage(content="hello")]
    async def run(model, config):
        return await model.ainvoke(messages, config) if asynchronous else model.invoke(messages, config)
    expected = await run(baseline, {})
    actual = await run(model, {"callbacks": [collector]})
    assert actual.model_dump(exclude={"id"}) == expected.model_dump(exclude={"id"})
    messages.extend([actual, HumanMessage(content="again")])
    await run(model, {"callbacks": [collector]})
    messages[1] = expected
    await run(baseline, {})
    assert requests == baseline_requests
    assert collector.summary["total_tokens"] == 50
    record = collector.records[0]
    assert ("usage_evidence" in record) is capture_evidence
    if capture_evidence:
        assert record["raw_usage"] == usage


@pytest.mark.parametrize("usage,expected", [
    ({"completion_tokens": 7}, (None, 7, None)),
    ({"prompt_tokens": 0, "completion_tokens": 0}, (0, 0, 0)),
    ({}, (None, None, None)),
    (None, (None, None, None)),
])
def test_missing_is_not_adapter_zero(usage, expected):
    model, _ = client("openai", usage=usage)
    collector = UsageCollector("s", "t", capture_evidence=True)
    model.invoke("hello", {"callbacks": [collector]})
    counts = collector.records[0]["counts"]
    assert tuple(counts[k] for k in ("input_tokens", "output_tokens", "total_tokens")) == expected
    assert collector.summary["partial"] is (expected[0] is None)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_interrupted_anthropic_stream_retains_start(asynchronous):
    model, _ = client("anthropic", stream_events=events("anthropic")[:3], interrupted=True)
    collector = UsageCollector("s", "t", capture_evidence=True)
    with pytest.raises(httpx.ReadError, match="disconnected"):
        if asynchronous:
            async for _ in model.astream("hello", {"callbacks": [collector]}):
                pass
        else:
            list(model.stream("hello", {"callbacks": [collector]}))
    record, = collector.records
    assert record["counts"]["input_tokens"] == 10
    assert record["counts"]["total_tokens"] == 10
    assert record["complete"] is False and record["status"] == "failed"
    assert collector.summary["partial"] is True
    assert _active.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_batched_invocations_bind_each_run(provider, asynchronous):
    usage = {"input_tokens": 10, "output_tokens": 7} if provider == "anthropic" else {"prompt_tokens": 10, "completion_tokens": 7}
    model, _ = client(provider, usage=usage)
    collector = UsageCollector("s", "t", capture_evidence=True)
    prompts = [[HumanMessage(content=str(i))] for i in range(6)]
    if asynchronous:
        await model.agenerate(prompts, callbacks=[collector])
    else:
        model.generate(prompts, callbacks=[collector])
    assert collector.summary["total_tokens"] == 102
    assert len({r["call_id"] for r in collector.records}) == 6
    assert all(r["raw_usage"] == usage for r in collector.records)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_concurrent_streams_do_not_cross_calls(asynchronous):
    collector = UsageCollector("s", "t", capture_evidence=True)
    models = [client("anthropic", stream_events=events("anthropic", inputs=i))[0] for i in range(1, 7)]
    config = {"callbacks": [collector]}
    if asynchronous:
        async def run(model):
            return [c async for c in model.astream("hello", config)]
        await asyncio.gather(*(run(m) for m in models))
    else:
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(lambda m: list(m.stream("hello", config)), models))
    assert sorted(r["counts"]["input_tokens"] for r in collector.records) == list(range(1, 7))
    assert collector.summary["total_tokens"] == 63
    assert _active.get() is None


def test_interleaved_streams_and_close_cleanup():
    collector = UsageCollector("s", "t", capture_evidence=True)
    streams = [client("anthropic", stream_events=events("anthropic", inputs=i))[0].stream(
        "hello", {"callbacks": [collector]}) for i in (10, 20)]
    for stream in streams:
        next(stream)
        assert _active.get() is None
    streams[0].close()
    list(streams[1])
    assert sorted(r["counts"]["input_tokens"] for r in collector.records) == [10, 20]
    assert collector.summary["partial"] is True
    assert _active.get() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["openai", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invoke_with_streaming_uses_same_capture(provider, asynchronous):
    model, _ = client(provider, stream_events=events(provider))
    model.streaming = True
    collector = UsageCollector("s", "t")
    if asynchronous:
        await model.ainvoke("hello", {"callbacks": [collector]})
    else:
        model.invoke("hello", {"callbacks": [collector]})
    assert collector.summary["total_tokens"] == 17
    assert collector.summary["partial"] is False


@pytest.mark.asyncio
async def test_cancelled_async_stream_cleans_up_after_start():
    model, _ = client("anthropic", stream_events=events("anthropic"))
    collector = UsageCollector("s", "t")
    stream = model.astream("hello", {"callbacks": [collector]})
    await anext(stream)
    await stream.aclose()
    assert _active.get() is None
    record, = collector.records
    assert record["counts"]["input_tokens"] == 10
    assert record["complete"] is False
    assert record["status"] == "failed"


def test_cumulative_anthropic_cache_and_null_updates():
    updates = events("anthropic")
    updates[0]["message"]["usage"].update(cache_read_input_tokens=20, cache_creation_input_tokens=5)
    updates[-2]["usage"].update(input_tokens=None, cache_read_input_tokens=None)
    updates.insert(-1, updates[-2])  # A repeated cumulative delta is not another bill.
    model, _ = client("anthropic", stream_events=updates)
    snapshots = []
    collector = UsageCollector("s", "t", capture_evidence=True, usage_sink=snapshots.append)
    list(model.stream("hello", {"callbacks": [collector]}))
    record, = collector.records
    assert record["counts"]["input_tokens"] == 35
    assert record["counts"]["output_tokens"] == 7
    assert record["counts"]["total_tokens"] == 42
    assert record["counts"]["cache_creation_input_tokens"] == 5
    assert snapshots[1]["counts"]["input_tokens"] == 35
    assert snapshots[1]["complete"] is False
    assert snapshots[-1]["complete"] is True
    assert record["usage_evidence"][-2]["usage"]["input_tokens"] is None


def test_responses_api_is_explicitly_unverified():
    response = {"id": "resp_test", "object": "response", "created_at": 1,
                "model": "gpt-test", "status": "completed", "error": None,
                "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text",
                            "text": "hello", "annotations": []}]}],
                "usage": {"input_tokens": 10, "output_tokens": 7, "total_tokens": 17,
                          "input_tokens_details": {"cached_tokens": 0},
                          "output_tokens_details": {"reasoning_tokens": 0}}}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    model = openai_compatible.build(dict(model="gpt-test", api_key="test-key",
                                         use_responses_api=True, max_retries=0,
                                         http_client=httpx.Client(transport=transport)))
    collector = UsageCollector("s", "t", capture_evidence=True)
    model.invoke("hello", {"callbacks": [collector]})
    record, = collector.records
    assert record["counts"]["total_tokens"] == 17
    assert record["source"] == "adapter_derived"
    assert record["raw_capture"] == "unverified"
    assert "usage_evidence" not in record


def test_native_v2_protocol_stream_is_explicitly_unverified():
    model, _ = client("anthropic", stream_events=events("anthropic"))
    collector = UsageCollector("s", "t", capture_evidence=True)
    stream = model.stream_events("hello", {"callbacks": [collector]}, version="v3")
    list(stream)
    record, = collector.records
    assert record["raw_capture"] == "unverified"
    assert "usage_evidence" not in record


@pytest.mark.parametrize("interrupted", [False, True])
def test_openai_cumulative_updates_and_interrupted_usage(interrupted):
    updates = events("openai")
    updates[0]["usage"] = {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}
    if interrupted:
        updates = updates[:1]
    else:
        updates.insert(-1, updates[0])
    model, _ = client("openai", stream_events=updates, interrupted=interrupted)
    collector = UsageCollector("s", "t", capture_evidence=True)
    if interrupted:
        with pytest.raises(httpx.ReadError):
            list(model.stream("hello", {"callbacks": [collector]}))
    else:
        list(model.stream("hello", {"callbacks": [collector]}))
    assert collector.summary["total_tokens"] == (11 if interrupted else 17)
    assert collector.summary["partial"] is interrupted
    assert collector.summary["call_count"] == 1
    assert _active.get() is None


def test_parse_error_keeps_usage_and_original_error():
    response = body("openai", {"prompt_tokens": 10, "completion_tokens": 7})
    response["choices"] = None
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    model = openai_compatible.build(dict(model="gpt-test", api_key="test-key", max_retries=0,
                                         http_client=httpx.Client(transport=transport)))
    collector = UsageCollector("s", "t", capture_evidence=True)
    with pytest.raises(TypeError, match="null value for 'choices'"):
        model.invoke("hello", {"callbacks": [collector]})
    assert collector.summary["total_tokens"] == 17
    assert collector.records[0]["status"] == "failed"
    assert _active.get() is None


@pytest.mark.asyncio
async def test_task_cancellation_retains_received_usage(monkeypatch):
    received = asyncio.Event()
    async def blocked(self):
        yield self.content
        received.set()
        await asyncio.Event().wait()
    monkeypatch.setattr(WireStream, "__aiter__", blocked)
    model, _ = client("anthropic", stream_events=events("anthropic")[:3])
    collector = UsageCollector("s", "t", capture_evidence=True)
    async def consume():
        async for _ in model.astream("hello", {"callbacks": [collector]}):
            pass
    task = asyncio.create_task(consume())
    await asyncio.wait_for(received.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert collector.summary["input_tokens"] == 10
    assert collector.summary["partial"] is True
    assert collector.records[0]["status"] == "failed"
    assert _active.get() is None
