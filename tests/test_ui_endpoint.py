"""Local endpoint probes validate URLs without blocking remote providers."""

from contextlib import nullcontext
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, build_opener
from urllib.response import addinfourl

import pytest

from ui import endpoint


@pytest.fixture(autouse=True)
def probe_opener(monkeypatch):
    """Isolate Streamlit caching and prevent real network requests."""
    endpoint.check_local_model_endpoint.clear()
    opener = Mock()
    opener.open.return_value = nullcontext(SimpleNamespace(status=200))
    monkeypatch.setattr(endpoint, "build_opener", Mock(return_value=opener))
    yield opener
    endpoint.check_local_model_endpoint.clear()


@pytest.mark.parametrize("base_url", [None, "", "   "])
def test_unconfigured_endpoint_skips_probe(base_url, probe_opener):
    result = endpoint.check_local_model_endpoint(base_url)

    assert result == {"ok": True, "message": "No base URL configured."}
    probe_opener.open.assert_not_called()


@pytest.mark.parametrize(
    "base_url",
    [
        "https://api.openai.com/v1",
        "https://apps.inside.anl.gov/argoapi/v1",
        "https://api.anthropic.com",
        "https://generativelanguage.googleapis.com/v1beta",
        "https://openrouter.ai/api/v1",
        "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
        "http://192.168.1.10:8000/v1",
    ],
)
def test_remote_endpoint_skips_probe(base_url, probe_opener):
    result = endpoint.check_local_model_endpoint(base_url)

    assert result == {"ok": True, "message": "Skipping non-local endpoint probe."}
    probe_opener.open.assert_not_called()


@pytest.mark.parametrize(
    ("base_url", "expected_url"),
    [
        ("http://localhost:11434", "http://localhost:11434/models"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1/models"),
        ("http://0.0.0.0:8000/v1", "http://0.0.0.0:8000/v1/models"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1/models"),
        ("https://[::1]/v1/", "https://[::1]/v1/models"),
        ("http://localhost:0", "http://localhost:0/models"),
        ("http://localhost:65535", "http://localhost:65535/models"),
        ("  HTTP://LOCALHOST/v1/  ", "http://localhost/v1/models"),
    ],
)
def test_local_endpoint_uses_canonical_probe_url(
    base_url, expected_url, probe_opener
):
    result = endpoint.check_local_model_endpoint(base_url)

    assert result == {"ok": True, "message": "Reachable (HTTP 200)."}
    request = probe_opener.open.call_args.args[0]
    assert request.full_url == expected_url
    assert request.get_method() == "GET"
    assert probe_opener.open.call_args.kwargs == {"timeout": 2}


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:bad/v1",
        "http://localhost:-1/v1",
        "http://localhost:65536/v1",
        "http://[::1]:bad/v1",
        "http://[::1]:65536/v1",
        "http://[::1/v1",
        "http://[invalid]/v1",
        "https://api.openai.com:bad/v1",
        "https://api.openai.com:65536/v1",
        "http:///v1",
        "localhost:8000/v1",
        "ftp://localhost/v1",
        "ftp://remote.example/v1",
        "file:///etc/hosts",
        "http://user:password@localhost/v1",
        "http://@localhost/v1",
        "http://localhost/v1?key=value",
        "http://localhost/v1#fragment",
    ],
)
def test_invalid_endpoint_returns_error_without_probing(base_url, probe_opener):
    result = endpoint.check_local_model_endpoint(base_url)

    assert result["ok"] is False
    assert "Invalid" in result["message"]
    probe_opener.open.assert_not_called()


@pytest.mark.parametrize("code", [401, 404, 500])
def test_http_error_still_reports_reachable(code, probe_opener):
    probe_opener.open.side_effect = HTTPError(
        "http://localhost:8000/models", code, "error", {}, None
    )

    result = endpoint.check_local_model_endpoint("http://localhost:8000")

    assert result == {"ok": True, "message": f"Reachable (HTTP {code})."}


@pytest.mark.parametrize("error", [URLError("connection refused"), TimeoutError()])
def test_connection_failure_returns_error(error, probe_opener):
    probe_opener.open.side_effect = error

    result = endpoint.check_local_model_endpoint("http://localhost:8000")

    assert result["ok"] is False
    assert result["message"].startswith("Unreachable:")


@pytest.mark.parametrize("code", [301, 302, 303, 307, 308])
def test_local_probe_does_not_follow_redirects(monkeypatch, code):
    """Exercise urllib's redirect handling with a fake HTTP transport."""
    requests = []

    class RedirectingHTTPHandler(HTTPHandler):
        def http_open(self, request):
            requests.append(request.full_url)
            headers = Message()
            headers["Location"] = "http://remote.example/models"
            response = addinfourl(BytesIO(b""), headers, request.full_url, code)
            response.msg = "Redirect"
            return response

    monkeypatch.setattr(
        endpoint,
        "build_opener",
        lambda *handlers: build_opener(RedirectingHTTPHandler(), *handlers),
    )

    result = endpoint.check_local_model_endpoint("http://localhost:8000/v1")

    assert result == {"ok": True, "message": f"Reachable (HTTP {code})."}
    assert requests == ["http://localhost:8000/v1/models"]
