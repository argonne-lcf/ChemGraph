#!/usr/bin/env python3
"""
Extract version information from pyproject.toml
Used by build scripts to get current version
"""

import sys
from pathlib import Path

def get_version():
    """Extract version from pyproject.toml"""
    # Get project root (parent of windows_packaging)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    
    pyproject_path = project_root / 'pyproject.toml'
    
    if not pyproject_path.exists():
        print("ERROR: pyproject.toml not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Try tomllib first (Python 3.11+)
        import tomllib
        with open(pyproject_path, 'rb') as f:
            data = tomllib.load(f)
            version = data.get('project', {}).get('version', '1.0.0')
    except ImportError:
        # Fallback to toml package
        try:
            import toml
            data = toml.load(pyproject_path)
            version = data.get('project', {}).get('version', '1.0.0')
        except ImportError:
            print("ERROR: Need tomllib (Python 3.11+) or toml package", file=sys.stderr)
            sys.exit(1)
    
    return version

if __name__ == '__main__':
    print(get_version())
