"""Optional service entry point stays lazy in a core installation."""

import sys
from types import ModuleType

import pytest


def test_service_dispatch_is_lazy(monkeypatch):
    from chemgraph.cli import main

    module = ModuleType('chemgraph.service.cli')
    calls = []
    module.main = lambda argv: calls.append(argv)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sys, 'argv', ['chemgraph', 'service', 'doctor', '--config', 'service.toml'])
    main()
    assert calls == [['doctor', '--config', 'service.toml']]


def test_missing_service_extra_has_actionable_error(monkeypatch):
    import builtins
    from chemgraph.cli import main

    original = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == 'chemgraph.service.cli':
            raise ModuleNotFoundError('a2a')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', unavailable)
    monkeypatch.setattr(sys, 'argv', ['chemgraph', 'service', '--help'])
    with pytest.raises(SystemExit, match=r'chemgraph\[service\]'):
        main()
