# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for ChemGraph Windows standalone package
Following PyInstaller best practices for Windows distribution
"""

import sys
import os
from pathlib import Path

# Get the project root directory
# When running from windows_packaging/config, go up two levels
if os.path.exists('src'):
    # Running from project root
    project_root = Path('.')
else:
    # Running from windows_packaging/config directory
    project_root = Path('../..')

src_path = project_root / 'src'

# Read version from pyproject.toml
def get_version():
    """Extract version from pyproject.toml"""
    try:
        import tomllib
        with open(project_root / 'pyproject.toml', 'rb') as f:
            data = tomllib.load(f)
            return data.get('project', {}).get('version', '1.0.0')
    except (ImportError, FileNotFoundError, KeyError):
        # Fallback for Python < 3.11 or if tomllib not available
        try:
            import toml
            data = toml.load(project_root / 'pyproject.toml')
            return data.get('project', {}).get('version', '1.0.0')
        except:
            return '1.0.0'

version = get_version()

# Prepare data files list
datas = []
config_file = project_root / 'config.toml'
if config_file.exists():
    datas.append((str(config_file), '.'))

# Collect all submodules for better dynamic import handling
def collect_submodules(package_name):
    """Collect all submodules of a package"""
    try:
        import importlib
        import pkgutil
        package = importlib.import_module(package_name)
        modules = [package_name]
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=package.__path__,
            prefix=package.__name__ + '.',
            onerror=lambda x: None
        ):
            modules.append(modname)
        return modules
    except:
        return [package_name]

# Enhanced hidden imports with submodule collection
hiddenimports = [
    # Core ChemGraph modules
    'chemgraph',
    'chemgraph.agent',
    'chemgraph.agent.llm_agent',
    'chemgraph.models',
    'chemgraph.models.supported_models',
    'chemgraph.graphs',
    'chemgraph.prompt',
    'chemgraph.state',
    'chemgraph.tools',
    'chemgraph.utils',
    'chemgraph.ui',
    'chemgraph.ui.cli',
    
    # LangChain and LangGraph
    'langgraph',
    'langchain_openai',
    'langchain_ollama',
    'langchain_anthropic',
    'langchain_google_genai',
    'langchain_groq',
    'langchain_experimental',
    
    # Scientific computing
    'ase',
    'ase.calculators',
    'rdkit',
    'rdkit.Chem',
    'rdkit.Chem.AllChem',
    'pandas',
    'numpy',
    'pymatgen',
    'tblite',
    'mace_torch',
    
    # Rich terminal UI
    'rich',
    'rich.console',
    'rich.panel',
    'rich.table',
    'rich.text',
    'rich.progress',
    'rich.syntax',
    'rich.markdown',
    'rich.prompt',
    'rich.live',
    'rich.layout',
    'rich.align',
    
    # Other dependencies
    'toml',
    'pydantic',
    'pydantic.v1',
    'pubchempy',
    'pyppeteer',
    'numexpr',
    'deepdiff',
    'streamlit',
    'stmol',
    'ipython_genutils',
    'langsmith',
    
    # Additional modules that might be imported dynamically
    'yaml',
    'json',
    'urllib3',
    'requests',
    'httpx',
    'aiohttp',
]

# Add submodules for key packages
try:
    hiddenimports.extend(collect_submodules('chemgraph'))
    hiddenimports.extend(collect_submodules('langgraph'))
    hiddenimports.extend(collect_submodules('langchain_openai'))
except:
    pass

a = Analysis(
    [str(project_root / 'src' / 'ui' / 'cli.py')],
    pathex=[str(src_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # GUI frameworks not needed for CLI
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
        # Testing frameworks
        'pytest',
        'unittest',
        # Development tools
        'IPython',
        'jupyter',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='chemgraph',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for CLI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if you have one: 'path/to/icon.ico'
    version=None,  # Can add version info file: 'version_info.txt'
)
