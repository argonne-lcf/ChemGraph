"""Authenticated native recovery through real checkpoints and hermetic EMT."""

import asyncio
from pathlib import Path

import httpx
import pytest
pytest.importorskip("a2a")
from a2a.types import a2a_pb2 as a2a
from langchain_core.messages import AIMessage

from chemgraph.service.adapters import ChemGraphAdapter
from tests.service.conftest import TOKENS, running
from tests.service.test_operations import approve
from tests.service.test_science import call, model as model



def script_energy(model):
    model[:] = [
        call('load_tools', names=['save_atomsdata_to_file', 'run_ase']),
        call('save_atomsdata_to_file', atomsdata={
            'numbers': [1, 1], 'positions': [[0, 0, 0], [0, 0, 0.8]],
        }, fname='h2.xyz'),
        call('run_ase', params={'input_structure_file': 'h2.xyz',
                              'output_results_file': 'energy.json', 'driver': 'energy',
                              'calculator': {'calculator_type': 'emt'}}),
        AIMessage(content='Energy is ready.'),
    ]


async def energy_review(service, client, app):
    task = (await client.send('Hydrogen energy', blocking=True, message_id='energy')).task
    await approve(service, app, task.id)
    task = await service.wait(task.id, 'alice')
    assert task.status.state == a2a.TASK_STATE_INPUT_REQUIRED
    operation = next(op for op in await service.operations.list(task.id) if op['tool'] == 'run_ase')
    assert operation['state'] == 'approval'
    return task, operation


async def operator_post(app, path, body):
    async with httpx.AsyncClient(base_url='http://test', transport=httpx.ASGITransport(app),
                                headers={'Authorization': f"Bearer {TOKENS['operator']}"}) as client:
        return await client.post(path, json=body)


@pytest.mark.asyncio
async def test_exact_input_approval_and_owner_isolation(settings, model):
    script_energy(model)
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task, operation = await energy_review(service, client, app)
        snapshot = operation['prepared']['inputs'][0]
        Path(snapshot['source']).write_text('changed after review')
        await approve(service, app, task.id, operation['id'])
        finished = await service.wait(task.id, 'alice')
        assert finished.status.state == a2a.TASK_STATE_COMPLETED
        energy = next(artifact for artifact in finished.artifacts if artifact.name == 'energy.json')
        result = await client.http.get('/artifacts/' + energy.artifact_id)
        assert result.status_code == 200
        assert result.json()['success'] is True
        async with httpx.AsyncClient(base_url='http://test', transport=httpx.ASGITransport(app),
                                    headers={'Authorization': f"Bearer {TOKENS['bob']}"}) as bob:
            assert (await bob.get('/artifacts/' + energy.artifact_id)).status_code == 404
        assert (await client.send('Hydrogen energy', message_id='energy')).task.id == task.id


@pytest.mark.parametrize('fault', ['capture', 'result_persistence'])
@pytest.mark.asyncio
async def test_restart_finalizes_or_recovers_without_recalculation(settings, model, monkeypatch, fault):
    from chemgraph.execution import scoped
    from chemgraph.tools import ase_tools

    script_energy(model)
    invoked = []
    original = ase_tools.run_ase_core

    def calculate(params):
        result = original(params)
        invoked.append(result)
        return result

    monkeypatch.setattr(ase_tools, 'run_ase_core', calculate)
    capture = scoped._capture
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task, operation = await energy_review(service, client, app)
        if fault == 'capture':
            def broken(*args):
                raise OSError('artifact disk unavailable')
            monkeypatch.setattr(scoped, '_capture', broken)
        else:
            put = service.operations.put

            async def broken(record):
                if record['state'] == 'returned' and record['tool'] == 'run_ase':
                    raise OSError('returned result commit failed')
                await put(record)
            monkeypatch.setattr(service.operations, 'put', broken)
        await approve(service, app, task.id, operation['id'])
        interrupted = await service.wait(task.id, 'alice')
        assert interrupted.status.state == a2a.TASK_STATE_INPUT_REQUIRED
        saved = await service.operations.get(operation['id'])
        assert saved['state'] == ('returned' if fault == 'capture' else 'invoking')
        assert len(invoked) == 1
    monkeypatch.setattr(scoped, '_capture', capture)
    model[:] = [AIMessage(content='Recovered energy.')]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        if fault == 'result_persistence':
            saved = await service.operations.get(operation['id'])
            assert saved['state'] == 'uncertain'
            response = await operator_post(app, f"/admin/operations/{operation['id']}/reconcile", {
                'mode': 'result', 'evidence': 'Inspected completed energy.json and calculation log',
                'result': invoked[0],
            })
            assert response.status_code == 200, response.text
        finished = await asyncio.wait_for(service.wait(task.id, 'alice'), 20)
        assert finished.status.state == a2a.TASK_STATE_COMPLETED, finished
        assert any(artifact.name == 'energy.json' for artifact in finished.artifacts)
        assert len(invoked) == 1
        assert (await service.operations.get(operation['id']))['state'] == 'completed'


@pytest.mark.asyncio
async def test_mixed_decisions_and_blocked_dependency(settings, model):
    model[:] = [call('load_tools', names=['save_atomsdata_to_file', 'run_ase']),
                AIMessage(content='', tool_calls=[
                    {'id': 'create', 'name': 'save_atomsdata_to_file', 'args': {
                        'atomsdata': {'numbers': [1], 'positions': [[0, 0, 0]]}, 'fname': 'h.xyz'}},
                    {'id': 'dependent', 'name': 'run_ase', 'args': {'params': {
                        'input_structure_file': 'h.xyz', 'driver': 'energy',
                        'calculator': {'calculator_type': 'emt'}}}},
                ]), AIMessage(content='Structure created; energy needs a new proposal.')]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task = (await client.send('Create then calculate', blocking=True)).task
        operations = await service.operations.list(task.id)
        create = next(op for op in operations if op['tool'] == 'save_atomsdata_to_file')
        dependent = next(op for op in operations if op['tool'] == 'run_ase')
        assert dependent['state'] == 'blocked'
        await approve(service, app, task.id, create['id'])
        completed = await service.wait(task.id, 'alice')
        assert completed.status.state == a2a.TASK_STATE_COMPLETED
        assert len(completed.artifacts) == 1
        assert (await service.operations.get(dependent['id']))['state'] == 'blocked'


@pytest.mark.asyncio
async def test_authenticated_rejection_does_not_execute(settings, model):
    model[:] = [call('load_tools', names=['save_atomsdata_to_file']),
                AIMessage(content='', tool_calls=[
                    {'id': name, 'name': 'save_atomsdata_to_file', 'args': {
                        'atomsdata': {'numbers': [1], 'positions': [[0, 0, 0]]}, 'fname': name + '.xyz'}}
                    for name in ['approved', 'rejected']
                ]), AIMessage(content='One structure saved.')]
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task = (await client.send('Create two structures', blocking=True)).task
        operations = await service.operations.list(task.id)
        rejected = next(op for op in operations if op['call']['id'] == 'rejected')
        body = {'digest': rejected['digest'], 'interrupt_id': rejected['interrupt_id'], 'message': 'Do only the first'}
        path = f"/admin/operations/{rejected['id']}/reject"
        assert (await client.http.post(path, json=body)).status_code == 403
        assert (await operator_post(app, path, body)).status_code == 200
        approved = next(op for op in operations if op['call']['id'] == 'approved')
        await approve(service, app, task.id, approved['id'])
        result = await service.wait(task.id, 'alice')
        assert result.status.state == a2a.TASK_STATE_COMPLETED
        assert [a.name for a in result.artifacts] == ['approved.xyz']
        assert (await service.operations.get(rejected['id']))['state'] == 'rejected'


@pytest.mark.parametrize('decision', ['approve', 'reject'])
@pytest.mark.asyncio
async def test_confirmed_nonexecution_requires_new_decision(settings, model, monkeypatch, decision):
    from chemgraph.execution import scoped

    script_energy(model)
    original_files = scoped._files
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task, operation = await energy_review(service, client, app)

        def unavailable(*args):
            raise OSError('preflight file inspection failed')

        monkeypatch.setattr(scoped, '_files', unavailable)
        await approve(service, app, task.id, operation['id'])
        await service.wait(task.id, 'alice')
        failed = await service.operations.get(operation['id'])
        assert failed['state'] == 'nonexecution'
        monkeypatch.setattr(scoped, '_files', original_files)
        assert not (Path(operation['workspace']) / 'energy.json').exists()
        response = await operator_post(app, f"/admin/operations/{operation['id']}/reconcile", {
            'mode': 'nonexecution', 'evidence': 'Durable preflight failure proves no invocation',
        })
        assert response.status_code == 200, response.text
        renewed = await service.operations.get(operation['id'])
        assert renewed['approval'] is None
        assert renewed['decisions'][0]['digest'] == operation['digest']
        assert renewed['previous_preparations'][0] == operation['prepared']
        assert renewed['reconciliations'][-1]['mode'] == 'nonexecution'
        assert renewed['interrupt_id'] != operation['interrupt_id']
        stale = await operator_post(app, f"/admin/operations/{operation['id']}/approve", {
            'digest': operation['digest'], 'interrupt_id': operation['interrupt_id'],
        })
        assert stale.status_code == 409
        response = await operator_post(app, f"/admin/operations/{operation['id']}/{decision}", {
            'digest': renewed['digest'], 'interrupt_id': renewed['interrupt_id'],
        })
        assert response.status_code == 200, response.text
        finished = await service.wait(task.id, 'alice')
        assert finished.status.state == a2a.TASK_STATE_COMPLETED, finished
        assert (Path(operation['workspace']) / 'energy.json').exists() == (decision == 'approve')


@pytest.mark.asyncio
async def test_artifact_registration_retries_saved_outcome(settings, monkeypatch):
    from tests.service.conftest import ArtifactAdapter

    async with running(settings, ArtifactAdapter(settings)) as (service, client, app):
        register = service.register_artifact

        async def unavailable(*args):
            raise OSError('registration volume unavailable')

        monkeypatch.setattr(service, 'register_artifact', unavailable)
        task = (await client.send('valuable output', blocking=True)).task
        record = await service.store.get(task.id)
        assert record['phase'] == 'finalization'
        assert record['pending_outcome']['artifacts']
        monkeypatch.setattr(service, 'register_artifact', register)
        response = await operator_post(app, f'/admin/tasks/{task.id}/finalize', {})
        assert response.status_code == 200
        assert response.json()['phase'] == 'done'
        finished = await client.get(task.id)
        assert finished.status.state == a2a.TASK_STATE_COMPLETED
        assert len(finished.artifacts) == 1
        assert (await service.store.get(task.id)).get('pending_outcome') is None


@pytest.mark.asyncio
async def test_abandonment_retires_thread_without_claiming_cancellation(settings, model, monkeypatch):
    from chemgraph.tools import ase_tools

    script_energy(model)
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task, operation = await energy_review(service, client, app)

        def lost(params):
            raise OSError('invocation acknowledgement lost')

        monkeypatch.setattr(ase_tools, 'run_ase_core', lost)
        await approve(service, app, task.id, operation['id'])
        await service.wait(task.id, 'alice')
        response = await operator_post(app, f"/admin/operations/{operation['id']}/reconcile", {
            'mode': 'abandon', 'evidence': 'Retiring this unresolved invocation after inspection',
        })
        assert response.status_code == 200, response.text
        task = await client.get(task.id)
        assert task.status.state == a2a.TASK_STATE_FAILED
        assert 'not confirmed' in task.status.message.parts[0].text
        assert await service.store.one('SELECT * FROM contexts WHERE context_id=?', (task.context_id,)) is None


@pytest.mark.asyncio
async def test_reconciliation_cannot_race_active_invocation(settings, model, monkeypatch):
    from dataclasses import replace

    script_energy(model)
    entered, release = asyncio.Event(), asyncio.Event()
    async with running(settings, ChemGraphAdapter(settings)) as (service, client, app):
        task, operation = await energy_review(service, client, app)
        execute = service.operations.execute_tool

        async def gated(record, context, call, stages):
            invoke = stages.invoke

            async def hold(preflight):
                entered.set()
                await release.wait()
                return await invoke(preflight)

            return await execute(record, context, call, replace(stages, invoke=hold))

        monkeypatch.setattr(service.operations, 'execute_tool', gated)
        await approve(service, app, task.id, operation['id'])
        await asyncio.wait_for(entered.wait(), 10)
        try:
            response = await operator_post(app, f"/admin/operations/{operation['id']}/reconcile", {
                'mode': 'nonexecution', 'evidence': 'This claim must not race active work',
            })
            assert response.status_code == 409
            assert 'active worker' in response.text
        finally:
            release.set()
        assert (await service.wait(task.id, 'alice')).status.state == a2a.TASK_STATE_COMPLETED
