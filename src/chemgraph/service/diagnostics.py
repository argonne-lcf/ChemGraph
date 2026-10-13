"""Installed-package and storage diagnostics; no Git-origin requirement."""

import sqlite3
import sys
from importlib.metadata import version
from urllib.parse import quote

from chemgraph.execution.scoped import EXECUTION_CONTRACT_VERSION


def diagnose(settings):
    settings.credentials()
    issues = []
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        issues.append('Service is tested on Python 3.11–3.13')
    if sys.platform == 'win32':
        issues.append('The single-instance service currently requires a POSIX host')
    stores = {}
    for name in ('service.db', 'a2a.db', 'checkpoints.db', 'sessions.db'):
        path = settings.state_dir / name
        if not path.exists():
            stores[name] = 'not created yet'
            continue
        with sqlite3.connect(f'file:{quote(str(path))}?mode=ro', uri=True) as connection:
            result = connection.execute('PRAGMA quick_check').fetchone()[0]
        stores[name] = result
        if result != 'ok':
            issues.append(f'{name} did not pass SQLite quick_check')
    return {
        'configured': not issues, 'python': sys.version.split()[0],
        'chemgraph': version('chemgraph'), 'a2a_sdk': version('a2a-sdk'),
        'execution_contract': EXECUTION_CONTRACT_VERSION,
        'profile': settings.profile.identity, 'adapter': settings.profile.adapter,
        'state_dir': str(settings.state_dir), 'stores': stores, 'issues': issues,
        'capabilities': settings.profile.capabilities, 'calculators': settings.profile.calculators,
    }
