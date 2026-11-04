"""Exercise pinned provider adapters with real SDKs and offline HTTP responses."""

import importlib
import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from chemgraph.models.loader import load_chat_model_prepared


@tool
def lookup(value: int) -> int:
    """Look up a test value."""
    return value


@pytest.mark.parametrize("provider,model_name", [
    ("openai", "gpt-4o"),
    ("anthropic", "claude-3-5-sonnet-20241022"),
    ("google", "gemini-2.5-pro"),
])
def test_provider_tool_and_usage_conversion(monkeypatch, provider, model_name):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    payloads = {
        "openai": {
            "id": "test", "object": "chat.completion", "created": 0,
            "model": model_name,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "", "tool_calls": [{
                    "id": "call-one", "type": "function", "function": {
                        "name": "lookup", "arguments": '{"value": 2}',
                    },
                }],
            }}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
        "anthropic": {
            "id": "msg-one", "type": "message", "role": "assistant",
            "model": model_name, "content": [{
                "type": "tool_use", "id": "call-one", "name": "lookup",
                "input": {"value": 2},
            }],
            "stop_reason": "tool_use", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        },
        "google": {
            "candidates": [{"content": {"role": "model", "parts": [{
                "functionCall": {"name": "lookup", "args": {"value": 2}},
            }]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 3, "candidatesTokenCount": 2, "totalTokenCount": 5,
            },
            "modelVersion": model_name,
        },
    }
    requests = []

    def send(_client, request, **kwargs):
        requests.append(json.loads(request.read()))
        http = importlib.import_module(type(request).__module__.split(".")[0])
        return http.Response(200, json=payloads[provider], request=request)

    # The SDKs can use either HTTPX namespace; intercept both before construction.
    for name in ("httpx", "httpx2"):
        http = importlib.import_module(name)
        monkeypatch.setattr(http.Client, "send", send)

    model, _ = load_chat_model_prepared(model_name=model_name, api_key="offline-test-key")
    model.streaming = False
    response = model.bind_tools([lookup]).invoke([
        SystemMessage(content="Use the lookup tool."), HumanMessage(content="Look up 2."),
    ])

    assert len(requests) == 1
    assert "lookup" in json.dumps(requests[0]["tools"])
    assert "Look up 2." in json.dumps(requests[0])
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0]["name"] == "lookup"
    assert response.tool_calls[0]["args"] == {"value": 2}
    assert response.usage_metadata["input_tokens"] == 3
    assert response.usage_metadata["output_tokens"] == 2
    assert response.usage_metadata["total_tokens"] == 5
