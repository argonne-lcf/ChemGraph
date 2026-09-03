"""Tests for Globus Transfer authentication and token persistence."""

from __future__ import annotations

import json
import stat
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from chemgraph.execution.globus_transfer import (
    TRANSFER_RESOURCE_SERVER,
    GlobusTransferManager,
)


def _manager(*, allow_interactive_auth: bool = True) -> GlobusTransferManager:
    return GlobusTransferManager(
        source_endpoint_id="source-id",
        destination_endpoint_id="destination-id",
        destination_base_path="/remote/staging",
        allow_interactive_auth=allow_interactive_auth,
    )


def _tokens(access_token: str = "access-token") -> dict:
    return {
        "access_token": access_token,
        "refresh_token": "refresh-token",
        "expires_at_seconds": 4_102_444_800,
    }


def _mock_sdk(authorizer: MagicMock | None = None):
    sdk = MagicMock()
    auth_client = MagicMock()
    sdk.NativeAppAuthClient.return_value = auth_client
    sdk.RefreshTokenAuthorizer.return_value = authorizer or MagicMock()
    sdk.TransferClient.return_value = MagicMock()
    return sdk, auth_client


def _token_path(tmp_path):
    return tmp_path / ".globus" / "chemgraph_transfer_tokens.json"


def test_cached_tokens_use_refresh_authorizer(tmp_path):
    token_path = _token_path(tmp_path)
    GlobusTransferManager._save_tokens(token_path, _tokens())
    authorizer = MagicMock()
    sdk, _ = _mock_sdk(authorizer)

    with (
        patch.dict(sys.modules, {"globus_sdk": sdk}),
        patch("chemgraph.execution.globus_transfer.Path.home", return_value=tmp_path),
    ):
        manager = _manager()
        manager.authenticate()

    sdk.RefreshTokenAuthorizer.assert_called_once()
    args, kwargs = sdk.RefreshTokenAuthorizer.call_args
    assert args == ("refresh-token", sdk.NativeAppAuthClient.return_value)
    assert kwargs["access_token"] == "access-token"
    assert kwargs["expires_at"] == 4_102_444_800
    assert callable(kwargs["on_refresh"])
    authorizer.get_authorization_header.assert_called_once_with()
    sdk.TransferClient.assert_called_once_with(authorizer=authorizer)


def test_refresh_callback_persists_rotated_access_token(tmp_path):
    token_path = _token_path(tmp_path)
    GlobusTransferManager._save_tokens(token_path, _tokens("old-access"))
    refresh_response = SimpleNamespace(
        by_resource_server={
            TRANSFER_RESOURCE_SERVER: {
                "access_token": "new-access",
                "expires_at_seconds": 4_200_000_000,
            }
        }
    )

    def make_authorizer(*_args, **kwargs):
        authorizer = MagicMock()
        authorizer.get_authorization_header.side_effect = lambda: kwargs[
            "on_refresh"
        ](refresh_response)
        return authorizer

    sdk, _ = _mock_sdk()
    sdk.RefreshTokenAuthorizer.side_effect = make_authorizer

    with (
        patch.dict(sys.modules, {"globus_sdk": sdk}),
        patch("chemgraph.execution.globus_transfer.Path.home", return_value=tmp_path),
    ):
        _manager().authenticate()

    stored = json.loads(token_path.read_text(encoding="utf-8"))
    assert stored == {
        "access_token": "new-access",
        "expires_at_seconds": 4_200_000_000,
        "refresh_token": "refresh-token",
    }
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_interactive_login_creates_restricted_token_cache(tmp_path):
    sdk, auth_client = _mock_sdk()
    auth_client.oauth2_get_authorize_url.return_value = "https://auth.example/login"
    auth_client.oauth2_exchange_code_for_tokens.return_value = SimpleNamespace(
        by_resource_server={TRANSFER_RESOURCE_SERVER: _tokens()}
    )

    with (
        patch.dict(sys.modules, {"globus_sdk": sdk}),
        patch("chemgraph.execution.globus_transfer.Path.home", return_value=tmp_path),
        patch("builtins.input", return_value="authorization-code") as read_code,
        patch("builtins.print") as print_message,
    ):
        _manager().authenticate()

    auth_client.oauth2_start_flow.assert_called_once_with(
        requested_scopes="urn:globus:auth:scope:transfer.api.globus.org:all",
        refresh_tokens=True,
    )
    auth_client.oauth2_exchange_code_for_tokens.assert_called_once_with(
        "authorization-code"
    )
    read_code.assert_called_once()
    print_message.assert_called_once()
    token_path = _token_path(tmp_path)
    assert json.loads(token_path.read_text(encoding="utf-8")) == _tokens()
    assert stat.S_IMODE(token_path.stat().st_mode) == 0o600


def test_noninteractive_missing_cache_never_reads_or_prints_stdio(tmp_path):
    sdk, _ = _mock_sdk()

    with (
        patch.dict(sys.modules, {"globus_sdk": sdk}),
        patch("chemgraph.execution.globus_transfer.Path.home", return_value=tmp_path),
        patch("builtins.input") as read_code,
        patch("builtins.print") as print_message,
        pytest.raises(RuntimeError, match="interactive authentication is disabled"),
    ):
        _manager(allow_interactive_auth=False).authenticate()

    read_code.assert_not_called()
    print_message.assert_not_called()
    sdk.TransferClient.assert_not_called()


def test_noninteractive_refresh_failure_never_falls_back_to_login(tmp_path):
    token_path = _token_path(tmp_path)
    GlobusTransferManager._save_tokens(token_path, _tokens())
    authorizer = MagicMock()
    authorizer.get_authorization_header.side_effect = RuntimeError("revoked")
    sdk, auth_client = _mock_sdk(authorizer)

    with (
        patch.dict(sys.modules, {"globus_sdk": sdk}),
        patch("chemgraph.execution.globus_transfer.Path.home", return_value=tmp_path),
        patch("builtins.input") as read_code,
        patch("builtins.print") as print_message,
        pytest.raises(RuntimeError, match="non-interactive process"),
    ):
        _manager(allow_interactive_auth=False).authenticate()

    read_code.assert_not_called()
    print_message.assert_not_called()
    auth_client.oauth2_start_flow.assert_not_called()


def test_interactive_refresh_failure_reauthenticates(tmp_path):
    token_path = _token_path(tmp_path)
    GlobusTransferManager._save_tokens(token_path, _tokens("revoked-access"))
    bad_authorizer = MagicMock()
    bad_authorizer.get_authorization_header.side_effect = RuntimeError("revoked")
    good_authorizer = MagicMock()
    sdk, auth_client = _mock_sdk()
    sdk.RefreshTokenAuthorizer.side_effect = [bad_authorizer, good_authorizer]
    auth_client.oauth2_get_authorize_url.return_value = "https://auth.example/login"
    auth_client.oauth2_exchange_code_for_tokens.return_value = SimpleNamespace(
        by_resource_server={TRANSFER_RESOURCE_SERVER: _tokens("replacement")}
    )

    with (
        patch.dict(sys.modules, {"globus_sdk": sdk}),
        patch("chemgraph.execution.globus_transfer.Path.home", return_value=tmp_path),
        patch("builtins.input", return_value="new-code"),
        patch("builtins.print"),
    ):
        _manager().authenticate()

    assert sdk.RefreshTokenAuthorizer.call_count == 2
    auth_client.oauth2_exchange_code_for_tokens.assert_called_once_with("new-code")
    good_authorizer.get_authorization_header.assert_called_once_with()
    assert json.loads(token_path.read_text(encoding="utf-8"))["access_token"] == (
        "replacement"
    )


def test_transfer_factory_forwards_noninteractive_auth_policy():
    from chemgraph.execution.config import get_transfer_manager

    manager = get_transfer_manager(
        source_endpoint_id="source-id",
        destination_endpoint_id="destination-id",
        destination_base_path="/remote/staging",
        allow_interactive_auth=False,
    )

    assert manager is not None
    assert manager.allow_interactive_auth is False
