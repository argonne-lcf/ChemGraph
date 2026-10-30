import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

from chemgraph.utils import orca


def executable(tmp_path, body, name='fake orca'):
    path = tmp_path / name
    path.write_text(f'#!{sys.executable}\n' + body)
    path.chmod(0o755)
    return path


def test_raw_path_spaces_env_precedence_and_no_shell(tmp_path, monkeypatch):
    path = executable(tmp_path, 'print("Program Version 6.0.0", flush=True)\n')
    monkeypatch.setenv('ASE_ORCA_COMMAND', str(path))
    assert orca.resolve_orca() == str(path)
    from chemgraph.schemas.calculators.orca_calc import OrcaCalc

    calc = OrcaCalc().get_calculator()
    assert calc.profile.get_command('orca.inp', None) == [str(path), 'orca.inp']
    for bad in [f'"{path}"', str(path) + ' | cat', '', 'mpirun -n 2 orca']:
        monkeypatch.setenv('ASE_ORCA_COMMAND', bad)
        with pytest.raises(FileNotFoundError):
            orca.resolve_orca()
        assert not orca.orca_available()


def test_common_path_and_path_discovery(monkeypatch):
    monkeypatch.delenv('ASE_ORCA_COMMAND', raising=False)
    seen = []

    def resolve(candidate):
        seen.append(candidate)
        if candidate != '/usr/local/orca/orca':
            raise FileNotFoundError(candidate)
        return candidate

    monkeypatch.setattr(orca, 'resolve_executable', resolve)
    assert orca.resolve_orca() == '/usr/local/orca/orca'
    assert seen == ['orca', '/opt/orca/orca', '/usr/local/orca/orca']


def test_identity_probe_cache_and_unrelated_program(tmp_path):
    path = executable(
        tmp_path, 'print("Program Version 6.0.0", flush=True)\nraise SystemExit(1)\n'
    )
    assert orca.identify_orca(str(path), tmp_path, orca.OrcaBudget()) == '6.0.0'
    assert orca.identify_orca(str(path), tmp_path, orca.OrcaBudget()) == '6.0.0'
    assert len(list(tmp_path.glob('identity-*'))) == 1
    path.write_text(f'#!{sys.executable}\nprint("Screen reader Orca 6.0.0")\n')
    with pytest.raises(ValueError, match='quantum-chemistry'):
        orca.identify_orca(str(path), tmp_path, orca.OrcaBudget())


@pytest.mark.parametrize(
    'setting', ['CHEMGRAPH_ORCA_TIMEOUT_S', 'CHEMGRAPH_ORCA_CALL_TIMEOUT_S']
)
@pytest.mark.parametrize('value', ['0', '-1', 'nan', '1.5', '86401', 'inf'])
def test_invalid_operator_budgets(monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError, match=setting):
        orca.OrcaBudget()


def test_actual_timeout_kills_descendants_and_preserves_diagnostics(tmp_path):
    marker = tmp_path / 'survived'
    code = f'import time; from pathlib import Path; time.sleep(2.5); Path({str(marker)!r}).write_text("bad")'
    path = executable(
        tmp_path,
        f'import subprocess,sys,time\nsubprocess.Popen([sys.executable,"-c",{code!r}])\nprint("diagnostic", flush=True)\ntime.sleep(20)\n',
    )
    output, error = tmp_path / 'out', tmp_path / 'err'
    with pytest.raises(TimeoutError):
        orca.run_bounded([str(path)], tmp_path, output, error, 1.5)
    assert output.read_text() == 'diagnostic\n'
    time.sleep(1.5)
    assert not marker.exists()


def test_probe_timeout_uses_remaining_call_budget(tmp_path):
    path = executable(tmp_path, 'import time\ntime.sleep(20)\n')
    budget = orca.OrcaBudget()
    budget.started -= budget.call_s - 0.1
    with pytest.raises(TimeoutError):
        orca.identify_orca(str(path), tmp_path, budget)
    assert list(tmp_path.glob('identity-*/probe.out'))
    with pytest.raises(TimeoutError):
        budget.remaining()


def test_legacy_configured_child(tmp_path):
    env = {
        **os.environ,
        'ASE_ORCA_COMMAND': shutil.which('false'),
        'CHEMGRAPH_ORCA_CHILD': '1',
        'PYTHONNOUSERSITE': '1',
        'PYTHONPATH': str(Path(__file__).parents[1] / 'src'),
        'CHEMGRAPH_LOG_DIR': str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_orca_legacy.py', '-q'],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_configured_schemas_execution_and_serialization_in_child(tmp_path):
    env = {
        **os.environ,
        'ASE_ORCA_COMMAND': shutil.which('false'),
        'CHEMGRAPH_ORCA_CHILD': '1',
        'PYTHONNOUSERSITE': '1',
        'PYTHONPATH': str(Path(__file__).parents[1] / 'src'),
        'CHEMGRAPH_LOG_DIR': str(tmp_path),
    }
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/test_orca_configured.py', '-q'],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
