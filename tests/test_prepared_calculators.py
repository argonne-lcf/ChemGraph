"""All calculator contracts; run this module without the not-tblite filter."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from chemgraph.execution.journal import DiskExecutionController
from chemgraph.execution.prepared import (
    CALCULATORS, ExecutionPolicy, PreparedAction, prepare_action, validate_preparation,
)
from chemgraph.execution.scoped import (
    ExecutionContext, ScopedExecutionError, execute_scoped_tool, execution_context,
)
from chemgraph.registry.tools import ToolSpec


def make_context(tmp_path, policy=None):
    context = ExecutionContext(
        'alice', 'thread', 'task', tmp_path / 'workspace',
        DiskExecutionController(tmp_path / 'journal'),
        model_roots=(tmp_path / 'resources',), policy=policy or ExecutionPolicy(),
    )
    (context.workspace / 'h2.xyz').write_text('2\nfixture\nH 0 0 0\nH 0 0 0.8\n')
    return context


def call(calc=None, driver='energy', identifier='energy'):
    return {'id': identifier, 'name': 'run_ase', 'args': {'params': {
        'input_structure_file': 'h2.xyz', 'output_results_file': 'out.json',
        'calculator': calc if calc is not None else {'calculator_type': 'emt'},
        'driver': driver,
    }}}


async def invoke(context, operation, handler):
    with execution_context(context):
        return await execute_scoped_tool(
            ToolSpec(operation['name'], '', None),
            SimpleNamespace(tool_call=operation), handler,
        )


@pytest.mark.parametrize('kind', sorted(CALCULATORS))
@pytest.mark.parametrize('driver', ['energy', 'dipole', 'opt', 'vib', 'ir', 'thermo'])
@pytest.mark.asyncio
async def test_every_calculator_and_driver(tmp_path, monkeypatch, kind, driver):
    from chemgraph.schemas import ase_input
    from chemgraph.utils import calculator_defaults

    monkeypatch.setattr(ase_input, '_engine_available', lambda name: True)
    monkeypatch.setattr(calculator_defaults, 'mace_polar_available', lambda: True)
    resources = tmp_path / 'resources'
    resources.mkdir()
    weights = resources / 'weights.pt'
    weights.write_bytes(b'operator-provisioned-weights')
    refs = resources / 'refs.json'
    refs.write_text('{"omol": {"1": 0.0}}')
    executable = resources / 'engine'
    executable.write_text('mock executable')
    settings = {
        'emt': {'asap_cutoff': True},
        'fairchem': {'spin': 3, 'charge': 1, 'task_name': 'omol', 'device': 'cpu',
                     'inference_settings': 'turbo', 'seed': 7},
        'tblite': {'method': 'GFN1-xTB', 'charge': 1, 'multiplicity': 2,
                   'accuracy': 0.1, 'max_iterations': 99, 'electronic_temperature': 400,
                   'electric_field': [0.1, 0, 0], 'mixer_damping': 0.3},
        'nwchem': {'basis': {'H': 'cc-pVDZ'}, 'theory': 'scf', 'multiplicity': 3,
                   'charge': 1, 'directory': 'nw'},
        'orca': {'charge': 1, 'multiplicity': 2, 'orcasimpleinput': 'PBE def2-SVP',
                 'orcablocks': '%pal nprocs 2 end', 'directory': 'orca'},
    }.get(kind, {})
    if kind.startswith('mace_'):
        settings = {'device': 'cpu', 'default_dtype': 'float32', 'charge': 1,
                    'multiplicity': 2, 'external_field': [0.1, 0.2, 0.3],
                    'dispersion': True, 'damping': 'zero', 'dispersion_xc': 'pbe'}
    model = 'uma-s-1p1' if kind == 'fairchem' else 'aimnet2' if kind == 'aimnet2' else 'default'
    policy = ExecutionPolicy(
        model_resources={f'{kind}:{model}': {'weights': str(weights), 'atom_refs': str(refs)}},
        executable_bindings={kind: {'command': str(executable), 'files': [str(executable)]}},
    )
    context = make_context(tmp_path, policy)
    operation = call({'calculator_type': kind, **settings}, driver)
    action = context.controller.prepare(context, operation)
    assert PreparedAction.from_dict(json.loads(json.dumps(action.to_dict()))) == action
    resolved = action.arguments['params']['calculator']
    for name, value in settings.items():
        name = 'multiplicity' if name == 'spin' else name
        if name == 'directory':
            value = str(context.workspace / value)
        assert resolved[name] == value
    assert action.arguments['params']['driver'] == driver
    if kind.startswith('mace_') or kind in {'fairchem', 'aimnet2'}:
        assert action.calculator['weights']['sha256']
        assert action.calculator['weights']['path'] == str(weights)
    if kind in {'orca', 'nwchem'}:
        assert action.calculator['executable']['command'] == str(executable)
    context.controller.decide(action, approve=True)

    async def handler(request):
        assert request.tool_call['args'] == action.arguments
        assert Path(request.tool_call['args']['params']['input_structure_file']).is_file()
        return ToolMessage(content='mock calculation', tool_call_id=operation['id'])

    assert (await invoke(context, operation, handler)).content == 'mock calculation'


def test_default_aliases_and_no_environment_selected_method(tmp_path, monkeypatch):
    from chemgraph.schemas import ase_input

    monkeypatch.setattr(ase_input, '_engine_available', lambda name: True)
    context = make_context(tmp_path)
    operation = call()
    operation['args']['params'].pop('calculator')
    with pytest.raises(ValueError, match='Clarification'):
        prepare_action(context, operation)
    context.policy = replace(context.policy, default_calculator={'calculator_type': 'emt'})
    assert prepare_action(context, operation).calculator['type'] == 'emt'
    action = prepare_action(context, call({'calculator_type': 'GFN1-xTB'}))
    assert action.arguments['params']['calculator']['method'] == 'GFN1-xTB'


def test_snapshot_and_resource_integrity(tmp_path):
    context = make_context(tmp_path)
    operation = call()
    action = prepare_action(context, operation)
    (context.workspace / 'h2.xyz').write_text('changed source')
    with execution_context(context):
        validate_preparation(context, operation, action)
    assert prepare_action(context, operation).digest != action.digest
    Path(action.inputs[0]['path']).write_text('tampered snapshot')
    with execution_context(context), pytest.raises(ValueError, match='changed'):
        validate_preparation(context, operation, action)
    altered = action.to_dict()
    altered['arguments']['params']['steps'] = 1
    with pytest.raises(ValueError, match='digest'):
        PreparedAction.from_dict(altered)


@pytest.mark.asyncio
async def test_real_emt_uses_snapshot_and_replays(tmp_path):
    from chemgraph.tools.ase_tools import run_ase

    context = make_context(tmp_path)
    operation = call()
    action = context.controller.prepare(context, operation)
    context.controller.decide(action, approve=True)
    (context.workspace / 'h2.xyz').write_text('changed source')
    calls = []

    async def handler(request):
        calls.append(1)
        return await run_ase.ainvoke({**request.tool_call, 'type': 'tool_call'})

    first = await invoke(context, operation, handler)
    assert json.loads(first.content)['status'] == 'success'
    context.controller = DiskExecutionController(tmp_path / 'journal')
    assert await invoke(context, operation, handler) == first
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('decision,expires', [(False, 600), (True, -1)])
async def test_rejected_expired_and_v1_never_execute(tmp_path, decision, expires):
    context = make_context(tmp_path)
    operation = call()
    action = context.controller.prepare(context, operation)
    context.controller.decide(action, approve=decision, expires_in=expires)

    async def handler(request):
        pytest.fail('unapproved invocation')

    with pytest.raises(ScopedExecutionError, match='approval'):
        await invoke(context, operation, handler)
    context.controller.contract_version = 1
    with pytest.raises(ScopedExecutionError, match='version 1'):
        await invoke(context, operation, handler)


def test_capability_missing_resources_and_limits(tmp_path, monkeypatch):
    from chemgraph.schemas import ase_input

    context = make_context(tmp_path)
    with pytest.raises(PermissionError, match='preparation'):
        prepare_action(context, {'id': 'x', 'name': 'run_xanes', 'args': {}})
    context.policy = replace(context.policy, calculators=frozenset())
    with pytest.raises(PermissionError, match='Calculator'):
        prepare_action(context, call())
    context.policy = ExecutionPolicy(limits={'max_steps': 2})
    with pytest.raises(PermissionError, match='step limit'):
        prepare_action(context, call())
    context.policy = ExecutionPolicy()
    monkeypatch.setattr(ase_input, '_engine_available', lambda name: False)
    with pytest.raises(ImportError, match='dependency'):
        prepare_action(context, call({'calculator_type': 'aimnet2'}))
    monkeypatch.setattr(ase_input, '_engine_available', lambda name: True)
    with pytest.raises(ValueError, match='Provision exact'):
        prepare_action(context, call({'calculator_type': 'aimnet2'}))


@pytest.mark.parametrize('kind', sorted(CALCULATORS))
def test_prepared_calculator_loaders_receive_exact_settings(tmp_path, monkeypatch, kind):
    from chemgraph.schemas import ase_input
    from chemgraph.tools.ase_core import load_calculator
    from chemgraph.execution.prepared import using_preparation
    from chemgraph.utils import calculator_defaults

    monkeypatch.setattr(ase_input, '_engine_available', lambda name: True)
    monkeypatch.setattr(calculator_defaults, 'mace_polar_available', lambda: True)
    root = tmp_path / 'resources'
    root.mkdir()
    weights = root / 'model.pt'
    weights.write_bytes(b'mock')
    refs = root / 'refs.json'
    refs.write_text('{}')
    model = 'uma-s-1p1' if kind == 'fairchem' else 'aimnet2' if kind == 'aimnet2' else 'default'
    context = make_context(tmp_path, ExecutionPolicy(
        model_resources={f'{kind}:{model}': {'weights': str(weights), 'atom_refs': str(refs)}},
        executable_bindings={kind: {'command': str(weights), 'files': [str(weights)]}},
    ))
    action = prepare_action(context, call({'calculator_type': kind}))
    schema = next(c for c in ase_input._known_calculator_classes
                  if (c.__name__ == 'MaceCalc' if kind.startswith('mace_')
                      else c().calculator_type.lower() == kind))
    seen = []

    def construct(self, **kwargs):
        seen.append((self, kwargs))
        return object()

    monkeypatch.setattr(schema, 'get_calculator', construct)
    with execution_context(context), using_preparation(action):
        _, properties, instance = load_calculator(action.arguments['params']['calculator'])
    assert seen[0][0] is instance
    if kind == 'orca':
        assert instance.profile.command == str(weights)
    elif kind == 'nwchem':
        assert instance.command == str(weights)
    elif kind == 'fairchem':
        assert seen[0][1] == {'checkpoint_path': str(weights), 'atom_refs_path': str(refs)}
        assert properties == {'charge': 0, 'spin': 1}
    elif kind.startswith('mace_') or kind == 'aimnet2':
        assert instance.model == str(weights)


@pytest.mark.parametrize('tool,arguments', [
    ('file_to_atomsdata', {'fname': 'h2.xyz'}),
    ('extract_output_json', {'json_file': 'result.json'}),
    ('smiles_to_atomsdata', {'smiles': 'O'}),
    ('smiles_to_coordinate_file', {'smiles': 'O'}),
    ('save_atomsdata_to_file', {'atomsdata': {'numbers': [1], 'positions': [[0, 0, 0]],
                                          'cell': [[0, 0, 0]] * 3, 'pbc': [False] * 3}}),
])
def test_structure_tools_resolve_defaults_and_snapshot_reads(tmp_path, tool, arguments):
    context = make_context(tmp_path)
    (context.workspace / 'result.json').write_text('{"energy":1}')
    action = prepare_action(context, {'id': 'structure', 'name': tool, 'args': arguments})
    if tool in {'file_to_atomsdata', 'extract_output_json'}:
        assert action.inputs[0]['sha256']
    elif tool == 'smiles_to_coordinate_file':
        assert action.arguments['output_file'] == str(context.workspace / 'molecule.xyz')
        assert action.arguments['randomSeed'] == 2025
    elif tool == 'save_atomsdata_to_file':
        assert action.arguments['fname'] == str(context.workspace / 'output.xyz')


@pytest.mark.asyncio
async def test_changed_model_resource_blocks_approved_invocation(tmp_path, monkeypatch):
    from chemgraph.schemas import ase_input

    monkeypatch.setattr(ase_input, '_engine_available', lambda name: True)
    root = tmp_path / 'resources'
    root.mkdir()
    model = root / 'aimnet2.pt'
    model.write_bytes(b'approved weights')
    context = make_context(tmp_path, ExecutionPolicy(
        model_resources={'aimnet2:aimnet2': str(model)},
    ))
    operation = call({'calculator_type': 'aimnet2'})
    action = context.controller.prepare(context, operation)
    context.controller.decide(action, approve=True)
    model.write_bytes(b'different weights')

    async def handler(request):
        pytest.fail('Changed weights must not reach the loader')

    with pytest.raises(ScopedExecutionError, match='resource changed'):
        await invoke(context, operation, handler)
    assert context.controller.get(action.operation_id)['state'] == 'nonexecution'
    assert prepare_action(context, operation).digest != action.digest
