#!/usr/bin/env python3
"""Refresh the Linux x86-64 / Python 3.11 evaluation dependency lock using uv."""

from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
output = ROOT / "evaluation/requirements.lock"
subprocess.run([
    "uv", "pip", "compile", str(ROOT / "pyproject.toml"), "--extra", "calculators",
    "--python-version", "3.11", "--python-platform", "x86_64-unknown-linux-gnu",
    "--torch-backend", "cpu", "--output-file", str(output), "--emit-index-url",
    "--custom-compile-command", "python evaluation/lock.py",
], check=True)
# uv's per-package torch index is not emitted as a pip index option.
text = output.read_text()
text = text.replace("--index-url https://pypi.org/simple",
                    "--index-url https://pypi.org/simple\n--extra-index-url https://download.pytorch.org/whl/cpu")
output.write_text(text)
