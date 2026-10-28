"""The production store preserves the journal's CAS and ownership contracts."""

import asyncio

import pytest

from chemgraph.runtime import OperationConflict, OperationJournal, SqliteOperationStore
from tests.runtime_helpers import make_context, run_ase_call, approve
from tests.test_runtime_journal import writer


@pytest.mark.asyncio
async def test_sqlite_journal_restart_and_owner_views(tmp_path):
    store = SqliteOperationStore(tmp_path / "operations.db")
    alice, bob = store.for_owner("alice"), store.for_owner("bob")
    context = make_context(tmp_path, journal=OperationJournal(alice))
    call = run_ase_call()
    record = await context.journal.propose(context, call)
    assert await bob.get(record["id"]) is None
    assert await bob.records(thread_id=context.thread_id) == []
    with pytest.raises(PermissionError):
        await bob.put(record, expected_revision=None)
    await approve(context, record)
    calls = []
    await context.journal.execute(context, call, writer(context, calls))
    context.journal = OperationJournal(SqliteOperationStore(store.path, owner="alice"))
    result = await context.journal.execute(context, call, writer(context, calls))
    assert len(calls) == 1 and result.content == "done"
    assert len(await alice.records(state="finalized")) == 1


@pytest.mark.asyncio
async def test_sqlite_compare_and_set_has_one_winner(tmp_path):
    store = SqliteOperationStore(tmp_path / "operations.db", owner="alice")
    initial = dict(id="one", owner="alice", thread_id="thread", state="approved", revision=1)
    await store.put(initial, expected_revision=None)
    attempts = [store.put({**initial, "revision": 2, "state": state}, expected_revision=1)
                for state in ("invoking", "abandoned")]
    results = await asyncio.gather(*attempts, return_exceptions=True)
    assert sum(isinstance(item, OperationConflict) for item in results) == 1
    with pytest.raises(OperationConflict):
        await store.put(initial, expected_revision=None)
    with pytest.raises(OperationConflict):
        await SqliteOperationStore(store.path).put(
            {**initial, "revision": 3, "owner": "bob"}, expected_revision=2)
