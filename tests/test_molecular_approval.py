import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from chemgraph.graphs.deep_agent import construct_deep_agent_graph
from chemgraph.registry import ToolRegistry
from tests.test_registry_middleware import CatalogModel, call, outputs


@pytest.mark.parametrize(
    'operation', ['prepare_conformers', 'calculate_reactivity_descriptors']
)
@pytest.mark.parametrize('decision', ['approve', 'reject'])
def test_molecular_tools_wait_for_approval(tmp_path, monkeypatch, operation, decision):
    from chemgraph.tools import molecular_core

    if operation == 'prepare_conformers':
        args = {
            'smiles': 'CN',
            'output_dir': str(tmp_path / 'runs'),
            'candidates': 2,
            'retained': 2,
        }
    else:
        args = {'result_file': str(tmp_path / 'result.json')}
    executed = []
    original = getattr(molecular_core, operation)

    def record_call(*a, **kw):
        executed.append(operation)
        return (
            original(*a, **kw)
            if operation == 'prepare_conformers'
            else {'success': True}
        )

    monkeypatch.setattr(molecular_core, operation, record_call)
    graph = construct_deep_agent_graph(
        CatalogModel(
            responses=[
                call('search_tools', query=operation),
                call('load_tools', names=[operation]),
                call(operation, **args),
                AIMessage(content='Done'),
            ]
        ),
        tool_registry=ToolRegistry(),
        discover_skills=False,
    )
    config = {'configurable': {'thread_id': operation}}
    state = graph.invoke(
        {'messages': [HumanMessage(content='Run the molecular tool.')]}, config
    )
    assert state['__interrupt__'] and not executed
    assert not (tmp_path / 'runs').exists()
    graph.invoke(Command(resume={'decisions': [{'type': decision}]}), config)
    assert executed == ([operation] if decision == 'approve' else [])
    assert (tmp_path / 'runs').exists() == (
        decision == 'approve' and operation == 'prepare_conformers'
    )


def test_discovery_approved_execution_and_artifact_retrieval(tmp_path):
    model = CatalogModel(
        responses=[
            call('search_tools', query='prepare conformers'),
            call('load_tools', names=['prepare_conformers', 'extract_output_json']),
            call(
                'prepare_conformers',
                smiles='CN',
                output_dir=str(tmp_path),
                candidates=2,
                retained=2,
            ),
            AIMessage(content='Prepared'),
        ]
    )
    graph = construct_deep_agent_graph(
        model, tool_registry=ToolRegistry(), discover_skills=False
    )
    config = {'configurable': {'thread_id': 'molecular-e2e'}}
    state = graph.invoke(
        {'messages': [HumanMessage(content='Prepare methylamine conformers.')]}, config
    )
    assert state['__interrupt__']
    assert not list(tmp_path.iterdir())
    state = graph.invoke(Command(resume={'decisions': [{'type': 'approve'}]}), config)
    result = json.loads(
        next(m.content for m in outputs(state) if m.name == 'prepare_conformers')
    )
    assert result['success'] and Path(result['manifest_path']).exists()
    model.responses.extend(
        [
            call('load_tools', names=['extract_output_json']),
            call('extract_output_json', json_file=result['manifest_path']),
            AIMessage(content='Retrieved'),
        ]
    )
    state = graph.invoke(
        {'messages': [HumanMessage(content='Retrieve the manifest.')]}, config
    )
    assert state['__interrupt__']
    state = graph.invoke(Command(resume={'decisions': [{'type': 'approve'}]}), config)
    saved = json.loads(
        next(m.content for m in outputs(state) if m.name == 'extract_output_json')
    )
    assert saved['molecule_id'] == result['molecule_id'] and saved['conformers']


@pytest.mark.parametrize(
    'name', ['prepare_conformers', 'calculate_reactivity_descriptors']
)
def test_explicit_approval_bypass_preserved(name, monkeypatch):
    from chemgraph.tools import molecular_core

    calls = []
    monkeypatch.setattr(
        molecular_core, name, lambda *a, **kw: calls.append(name) or {'success': True}
    )
    args = (
        {'smiles': 'CN'}
        if name == 'prepare_conformers'
        else {'result_file': 'saved.json'}
    )
    graph = construct_deep_agent_graph(
        CatalogModel(
            responses=[
                call('load_tools', names=[name]),
                call(name, **args),
                AIMessage(content='Done'),
            ]
        ),
        tool_registry=ToolRegistry(),
        discover_skills=False,
        interrupt_on=None,
    )
    state = graph.invoke(
        {'messages': [HumanMessage(content='Run it.')]},
        {'configurable': {'thread_id': name}},
    )
    assert '__interrupt__' not in state and calls == [name]


def test_scoped_denial_and_explicit_driver_before_snapshots(tmp_path):
    from chemgraph.runtime import PreparationError, PolicyDenied, run_context
    from chemgraph.runtime.policy import prepare
    from chemgraph.tools.molecular_core import (
        prepare_conformers,
        calculate_reactivity_descriptors,
    )
    from tests.runtime_helpers import make_context, run_ase_call

    context = make_context(tmp_path)
    for operation in ['prepare_conformers', 'calculate_reactivity_descriptors']:
        with pytest.raises(PolicyDenied):
            prepare(context, {'id': operation, 'name': operation, 'args': {}})
    for driver in [None, 'missing']:
        request = run_ase_call()
        if driver is None:
            request['args']['params']['driver'] = None
        else:
            request['args']['params'].pop('driver')
        with pytest.raises(PreparationError, match='explicit'):
            prepare(context, request)
    with pytest.raises((PolicyDenied, PreparationError)):
        prepare(context, run_ase_call(driver='electronic'))
    with pytest.raises(PolicyDenied):
        prepare(context, run_ase_call(calculator={'calculator_type': 'orca'}))
    assert not (context.workspace / '.chemgraph-inputs').exists()
    with run_context(context):
        with pytest.raises(PermissionError, match='unscoped'):
            prepare_conformers('CN', str(tmp_path / 'runs'))
        with pytest.raises(PermissionError, match='unscoped'):
            calculate_reactivity_descriptors('not-read.json')
