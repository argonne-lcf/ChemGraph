"""Transport contracts and authentication without service credentials."""

import json
import os
from pathlib import Path
from unittest.mock import Mock

import globus_sdk
import pytest

from chemgraph.execution import globus_transfer as transfer_module
from chemgraph.execution.globus_transfer import (
    GlobusTransferManager,
    TransferAuthenticationRequired,
)


def test_missing_transfer_auth_never_prompts(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(
        "builtins.input", Mock(side_effect=AssertionError("must not prompt"))
    )
    manager = GlobusTransferManager("source", "dest", "/remote")
    with pytest.raises(TransferAuthenticationRequired):
        manager._get_transfer_client()


@pytest.mark.parametrize("client_id", [None, "custom-client"])
def test_refresh_authorizer_updates_cache_and_survives_session(monkeypatch, tmp_path, client_id):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    expected_client = client_id or transfer_module._DEFAULT_CLIENT_ID
    cache = transfer_module._token_file(expected_client)
    tokens = {"access_token": "old", "refresh_token": "refresh", "expires_at_seconds": 1}
    if client_id:
        tokens["client_id"] = client_id
    else:
        assert cache == tmp_path / ".globus/chemgraph_transfer_tokens.json"
    GlobusTransferManager._save_tokens(cache, tokens)
    auth = Mock()
    auth.oauth2_refresh_token.return_value = Mock(
        by_resource_server={
            "transfer.api.globus.org": {
                "access_token": "new",
                "expires_at_seconds": 9999999999,
            },
        }
    )
    factory = Mock(return_value=auth)
    monkeypatch.setattr(globus_sdk, "NativeAppAuthClient", factory)
    manager = GlobusTransferManager("source", "dest", "/remote", client_id=client_id)
    client = manager._get_transfer_client()
    assert isinstance(client.authorizer, globus_sdk.RefreshTokenAuthorizer)
    assert client.authorizer.get_authorization_header() == "Bearer new"
    assert auth.oauth2_refresh_token.call_args.args == ("refresh",)
    assert json.loads(cache.read_text())["refresh_token"] == "refresh"
    assert json.loads(cache.read_text())["client_id"] == expected_client
    factory.assert_called_once_with(expected_client)
    if os.name == "posix":
        assert cache.stat().st_mode & 0o777 == 0o600
    client.authorizer.expires_at = 1
    client.authorizer.get_authorization_header()
    assert auth.oauth2_refresh_token.call_count == 2


@pytest.mark.parametrize("client_id", [None, "custom-client"])
def test_terminal_login_and_refresh_use_same_client_and_cache(monkeypatch, tmp_path, client_id):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "authorization-code")
    expected_client = client_id or transfer_module._DEFAULT_CLIENT_ID
    other_client = "other-client" if client_id is None else transfer_module._DEFAULT_CLIENT_ID
    other_cache = transfer_module._token_file(other_client)
    GlobusTransferManager._save_tokens(other_cache, {"refresh_token": "untouched"})
    original = other_cache.read_bytes()
    auth = Mock()
    auth.oauth2_get_authorize_url.return_value = "https://example.invalid/login"
    auth.oauth2_exchange_code_for_tokens.return_value = Mock(by_resource_server={
        "transfer.api.globus.org": {
            "access_token": "login-token", "refresh_token": "refresh", "expires_at_seconds": 1,
        },
    })
    auth.oauth2_refresh_token.return_value = Mock(by_resource_server={
        "transfer.api.globus.org": {"access_token": "new", "expires_at_seconds": 9999999999},
    })
    factory = Mock(return_value=auth)
    monkeypatch.setattr(globus_sdk, "NativeAppAuthClient", factory)
    args = ["--collection", "first-collection", "--collection", "second-collection"]
    if client_id:
        args += ["--client-id", client_id]
    transfer_module.main(args)
    scope = auth.oauth2_start_flow.call_args.kwargs["requested_scopes"]
    for collection in ("first-collection", "second-collection"):
        assert f"https://auth.globus.org/scopes/{collection}/data_access" in scope
    assert auth.oauth2_start_flow.call_args.kwargs["refresh_tokens"]
    auth.oauth2_exchange_code_for_tokens.assert_called_once_with("authorization-code")
    cache = transfer_module._token_file(expected_client)
    assert json.loads(cache.read_text())["client_id"] == expected_client
    manager = GlobusTransferManager("source", "dest", "/remote", client_id=client_id)
    assert manager._get_transfer_client().authorizer.get_authorization_header() == "Bearer new"
    assert [call.args for call in factory.call_args_list] == [(expected_client,), (expected_client,)]
    assert json.loads(cache.read_text())["refresh_token"] == "refresh"
    assert json.loads(cache.read_text())["client_id"] == expected_client
    if os.name == "posix":
        assert cache.stat().st_mode & 0o777 == 0o600
    assert other_cache.read_bytes() == original


@pytest.mark.parametrize("client_id, metadata", [
    (None, {"client_id": "other-client"}),
    (None, {"client_id": None}),
    ("custom-client", {"client_id": "other-client"}),
    ("custom-client", {}),
])
def test_mismatched_or_untagged_custom_cache_requires_login(monkeypatch, tmp_path, client_id, metadata):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = transfer_module._token_file(client_id or transfer_module._DEFAULT_CLIENT_ID)
    GlobusTransferManager._save_tokens(cache, {"refresh_token": "private-token", **metadata})
    original = cache.read_bytes()
    authorizer = Mock()
    monkeypatch.setattr(globus_sdk, "RefreshTokenAuthorizer", authorizer)
    manager = GlobusTransferManager("source", "dest", "/remote", client_id=client_id)
    with pytest.raises(TransferAuthenticationRequired) as error:
        manager._get_transfer_client()
    assert "python -m chemgraph.execution.globus_transfer" in str(error.value)
    if client_id:
        assert "--client-id custom-client" in str(error.value)
    assert "private-token" not in str(error.value)
    authorizer.assert_not_called()
    assert cache.read_bytes() == original


def test_custom_client_never_falls_back_to_legacy_shared_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = tmp_path / ".globus/chemgraph_transfer_tokens.json"
    GlobusTransferManager._save_tokens(cache, {"refresh_token": "legacy-token"})
    manager = GlobusTransferManager("source", "dest", "/remote", client_id="custom-client")
    with pytest.raises(TransferAuthenticationRequired, match="--client-id custom-client"):
        manager._get_transfer_client()
    assert json.loads(cache.read_text()) == {"refresh_token": "legacy-token"}


def test_explicit_mapping_and_legacy_layout(monkeypatch, tmp_path):
    manager = GlobusTransferManager("source", "dest", "/remote")
    manager._transfer_client = Mock()
    manager._transfer_client.get_submission_id.return_value = {"value": "submission"}
    manager._transfer_client.submit_transfer.return_value = {"task_id": "transfer"}
    paths = [str(tmp_path / "a/in.xyz"), str(tmp_path / "b/in.xyz")]
    result = manager.transfer_files(paths, remote_subdir="run")
    assert list(result.file_mapping.values()) == [
        "/remote/run/in.xyz",
        "/remote/run/in_1.xyz",
    ]
    client = manager._transfer_client
    payload = client.submit_transfer.call_args.args[0]
    assert isinstance(payload, globus_sdk.TransferData)
    assert [item["destination_path"] for item in payload["DATA"]] == list(result.file_mapping.values())
    mapping = {"/collection/a/in.xyz": "/remote/run/a/in.xyz"}
    prepared = manager.prepare_mapping(mapping)
    assert client.submit_transfer.call_count == 1
    mapping.clear()
    assert manager.submit_prepared(prepared) == "transfer"
    payload = client.submit_transfer.call_args.args[0]
    assert isinstance(payload, globus_sdk.TransferData)
    assert [(item["source_path"], item["destination_path"]) for item in payload["DATA"]] == [
        ("/collection/a/in.xyz", "/remote/run/a/in.xyz")
    ]
    assert payload["verify_checksum"] is True
    assert payload["sync_level"] == 3
    assert payload["label"] == "ChemGraph HPC staging"
    manager.transfer_mapping({"/remote/out": "/collection/out"}, reverse=True)
    payload = client.submit_transfer.call_args.args[0]
    assert (payload["source_endpoint"], payload["destination_endpoint"]) == ("dest", "source")
    assert "client" not in repr(prepared) and "payload" not in repr(prepared)


def _token_response(response_class, status=200, **fields):
    import requests

    raw = requests.Response()
    raw.status_code = status
    raw.url = "https://auth.globus.org/v2/oauth2/token"
    raw.request = requests.Request("POST", raw.url).prepare()
    raw.headers["Content-Type"] = "application/json"
    raw._content = json.dumps(fields).encode()
    if response_class is globus_sdk.AuthAPIError:
        return response_class(raw)
    return response_class(raw, client=Mock())


def _refresh_response(access_token="new", **fields):
    # Real responses report an omitted refresh token as None, not a missing key.
    return _token_response(
        globus_sdk.OAuthRefreshTokenResponse, access_token=access_token,
        expires_in=3600, resource_server="transfer.api.globus.org",
        scope=transfer_module.TRANSFER_SCOPE, token_type="Bearer", other_tokens=[],
        **fields,
    )


@pytest.mark.parametrize("cached", [
    {"access_token": "old", "refresh_token": "refresh", "expires_at_seconds": 1},
    {"refresh_token": "refresh"},
])
def test_refresh_without_replacement_token_keeps_refresh_token(monkeypatch, tmp_path, cached):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = transfer_module._token_file(transfer_module._DEFAULT_CLIENT_ID)
    GlobusTransferManager._save_tokens(cache, cached)
    auth = Mock()
    auth.oauth2_refresh_token.side_effect = lambda _: _refresh_response()
    monkeypatch.setattr(globus_sdk, "NativeAppAuthClient", Mock(return_value=auth))
    manager = GlobusTransferManager("source", "dest", "/remote")
    authorizer = manager._get_transfer_client().authorizer
    assert authorizer.get_authorization_header() == "Bearer new"
    authorizer.expires_at = 1
    authorizer.get_authorization_header()
    assert [call.args for call in auth.oauth2_refresh_token.call_args_list] == [
        ("refresh",), ("refresh",)
    ]
    assert authorizer.refresh_token == "refresh"
    assert json.loads(cache.read_text())["refresh_token"] == "refresh"


def test_terminal_login_replaces_running_manager_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "authorization-code")
    cache = transfer_module._token_file(transfer_module._DEFAULT_CLIENT_ID)
    GlobusTransferManager._save_tokens(cache, {
        "access_token": "stale", "refresh_token": "old-refresh", "expires_at_seconds": 1,
    })
    auth = Mock()
    auth.oauth2_refresh_token.side_effect = lambda _: _refresh_response()
    auth.oauth2_exchange_code_for_tokens.return_value = Mock(by_resource_server={
        "transfer.api.globus.org": {
            "access_token": "login-token", "refresh_token": "new-refresh",
            "expires_at_seconds": 9999999999,
        },
    })
    monkeypatch.setattr(globus_sdk, "NativeAppAuthClient", Mock(return_value=auth))
    manager = GlobusTransferManager("source", "dest", "/remote")
    stale_client = manager._get_transfer_client()
    transfer_module.authenticate()
    stale_client.authorizer.get_authorization_header()
    assert json.loads(cache.read_text())["refresh_token"] == "new-refresh"
    assert json.loads(cache.read_text())["access_token"] == "login-token"
    client = manager._get_transfer_client()
    assert client is not stale_client
    assert client.authorizer.refresh_token == "new-refresh"
    assert client.authorizer.get_authorization_header() == "Bearer login-token"
    assert manager._get_transfer_client() is client


def test_own_refresh_does_not_rebuild_client(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = transfer_module._token_file(transfer_module._DEFAULT_CLIENT_ID)
    GlobusTransferManager._save_tokens(cache, {
        "access_token": "old", "refresh_token": "refresh", "expires_at_seconds": 1,
    })
    auth = Mock()
    auth.oauth2_refresh_token.side_effect = lambda _: _refresh_response()
    monkeypatch.setattr(globus_sdk, "NativeAppAuthClient", Mock(return_value=auth))
    manager = GlobusTransferManager("source", "dest", "/remote")
    client = manager._get_transfer_client()
    client.authorizer.get_authorization_header()
    assert json.loads(cache.read_text())["access_token"] == "new"
    assert manager._get_transfer_client() is client


def test_refresh_errors_distinguish_login_from_network(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = transfer_module._token_file(transfer_module._DEFAULT_CLIENT_ID)
    GlobusTransferManager._save_tokens(cache, {
        "access_token": "old", "refresh_token": "refresh", "expires_at_seconds": 1,
    })
    auth = Mock()
    monkeypatch.setattr(globus_sdk, "NativeAppAuthClient", Mock(return_value=auth))
    manager = GlobusTransferManager("source", "dest", "/remote")
    auth.oauth2_refresh_token.side_effect = globus_sdk.GlobusConnectionError(
        "unreachable", Exception()
    )
    with pytest.raises(globus_sdk.GlobusConnectionError):
        manager.prepare_mapping({"/a": "/b"})
    auth.oauth2_refresh_token.side_effect = _token_response(
        globus_sdk.AuthAPIError, status=400, error="invalid_grant"
    )
    with pytest.raises(TransferAuthenticationRequired):
        manager.prepare_mapping({"/a": "/b"})


def test_unreadable_cache_requires_login(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    cache = transfer_module._token_file(transfer_module._DEFAULT_CLIENT_ID)
    cache.parent.mkdir()
    cache.write_bytes(b"\xff\xfe")
    with pytest.raises(TransferAuthenticationRequired):
        GlobusTransferManager("source", "dest", "/remote")._get_transfer_client()


@pytest.mark.parametrize("names, expected", [
    (["a/in.xyz", "b/in.xyz", "c/in_1.xyz"], ["in.xyz", "in_1.xyz", "in_1_1.xyz"]),
    (["c/in_1.xyz", "a/in.xyz", "b/in.xyz"], ["in_1.xyz", "in.xyz", "in_2.xyz"]),
    (["a/in.xyz", "a/in.xyz", "b/in"], ["in.xyz", "in"]),
])
def test_legacy_layout_allocates_unique_destinations(tmp_path, names, expected):
    manager = GlobusTransferManager("source", "dest", "/remote")
    manager._transfer_client = Mock()
    manager._transfer_client.submit_transfer.return_value = {"task_id": "transfer"}
    result = manager.transfer_files([str(tmp_path / name) for name in names], remote_subdir="run")
    destinations = [f"/remote/run/{name}" for name in expected]
    assert list(result.file_mapping.values()) == destinations
    payload = manager._transfer_client.submit_transfer.call_args.args[0]
    assert [item["destination_path"] for item in payload["DATA"]] == destinations
