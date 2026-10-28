"""Local token lifecycle and stable user identities."""

import time

import pytest

from chemgraph.mcp.service.auth import TokenStore, principal


def test_tokens_are_hashed_revocable_and_keep_stable_identity(tmp_path):
    store = TokenStore(tmp_path / "auth.db")
    first = store.issue("alice")
    alice = store.verify(first)
    second = store.rotate(alice.token_id)
    assert store.verify(first) is None
    assert store.verify(second).user_id == alice.user_id
    assert store.verify(store.issue("alice")).user_id == alice.user_id
    assert all("digest" not in row for row in store.list_tokens())
    assert first.encode() not in (tmp_path / "auth.db").read_bytes()
    assert store.revoke(store.verify(second).token_id)
    assert store.verify(second) is None
    expired = store.issue("bob", lifetime=0.001)
    time.sleep(0.005)
    assert store.verify(expired) is None
    with pytest.raises(PermissionError):
        principal()
