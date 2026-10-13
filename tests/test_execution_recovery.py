"""Fault injection at durable execution boundaries, with actual disk restart."""

from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.messages import ToolMessage

from chemgraph.execution import scoped
from chemgraph.execution.journal import DiskExecutionController
from chemgraph.execution.scoped import ScopedExecutionError
from tests.test_prepared_calculators import call, invoke, make_context


@pytest.mark.asyncio
async def test_preflight_failure_proves_nonexecution(tmp_path):
    context = make_context(tmp_path)
    operation = call()
    action = context.controller.prepare(context, operation)
    context.controller.decide(action, approve=True)
    Path(action.inputs[0]['path']).write_text('tampered')

    async def handler(request):
        pytest.fail('preflight must precede invocation')

    with pytest.raises(ScopedExecutionError, match='changed'):
        await invoke(context, operation, handler)
    assert context.controller.get(action.operation_id)['state'] == 'nonexecution'


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['invocation', 'persistence'])
async def test_uncertainty_survives_restart_and_blocks_workspace(tmp_path, monkeypatch, failure):
    context = make_context(tmp_path)
    operation = call()
    action = context.controller.prepare(context, operation)
    context.controller.decide(action, approve=True)
    original_put = context.controller.put
    calls = []

    def put(record):
        if record['state'] == 'returned':
            raise OSError('lost returned result')
        original_put(record)

    if failure == 'persistence':
        monkeypatch.setattr(context.controller, 'put', put)

    async def handler(request):
        calls.append(1)
        (context.workspace / 'output.txt').write_text('valuable completed work')
        if failure == 'invocation':
            raise OSError('lost acknowledgement')
        return ToolMessage(content='done', tool_call_id=operation['id'])

    with pytest.raises(ScopedExecutionError):
        await invoke(context, operation, handler)
    context.controller = DiskExecutionController(tmp_path / 'journal')
    assert context.controller.get(action.operation_id)['state'] == 'invoking'
    with pytest.raises(ScopedExecutionError, match='uncertain'):
        await invoke(context, operation, handler)
    other = call(identifier='other')
    next_action = context.controller.prepare(context, other)
    context.controller.decide(next_action, approve=True)
    with pytest.raises(ScopedExecutionError, match='unfinished'):
        await invoke(context, other, handler)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_finalization_retry_after_restart_preserves_failure_result(tmp_path, monkeypatch):
    context = make_context(tmp_path)
    operation = call()
    action = context.controller.prepare(context, operation)
    context.controller.decide(action, approve=True)
    capture = scoped._capture
    calls = []

    def unavailable(*args):
        raise OSError('artifact volume unavailable')

    monkeypatch.setattr(scoped, '_capture', unavailable)

    async def handler(request):
        calls.append(1)
        (context.workspace / 'scientific.json').write_text('{"converged":false}')
        return ToolMessage(content='scientific failure', tool_call_id=operation['id'])

    with pytest.raises(ScopedExecutionError, match='artifact'):
        await invoke(context, operation, handler)
    saved = context.controller.get(action.operation_id)
    assert saved['state'] == 'returned'
    assert saved['preflight']['baseline'] == {'h2.xyz': action.inputs[0]['sha256']}
    context.controller = DiskExecutionController(tmp_path / 'journal')
    other = call(identifier='other')
    with pytest.raises(ScopedExecutionError, match='unfinished'):
        await invoke(context, other, handler)
    monkeypatch.setattr(scoped, '_capture', capture)
    result = await invoke(context, operation, handler)
    assert result.content == 'scientific failure'
    artifact = result.additional_kwargs[scoped.ARTIFACT_KEY][0]
    assert Path(artifact['path']).read_text() == '{"converged":false}'
    assert len(calls) == 1
    assert context.controller.get(action.operation_id)['state'] == 'finalized'


@pytest.mark.asyncio
async def test_policy_change_blocks_execution(tmp_path):
    context = make_context(tmp_path)
    operation = call()
    action = context.controller.prepare(context, operation)
    context.controller.decide(action, approve=True)
    context.policy = replace(context.policy, configuration_id='changed')

    async def handler(request):
        pytest.fail('configuration changed after approval')

    with pytest.raises(ScopedExecutionError, match='policy'):
        await invoke(context, operation, handler)
