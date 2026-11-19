#!/usr/bin/env python3
"""Build only allowlisted source files in a temporary context, never the checkout."""

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def stage_context(destination):
    """Copy package sources and intentional package data; exclude benchmark data."""
    destination = Path(destination)
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copyfile(ROOT / name, destination / name)
    for path in (ROOT / "src").rglob("*"):
        relative = path.relative_to(ROOT)
        if (path.is_symlink() or not path.is_file() or "__pycache__" in path.parts
                or path.suffix not in {".py", ".md", ".template", ".def"}
                and path.name not in {"ocsr_registry.json", "simulation.input"}):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    for name in ("Worker.Containerfile", "Proxy.Containerfile", "squid.conf",
                 "prepare_image.py", "requirements.lock", "build.lock"):
        shutil.copyfile(ROOT / "evaluation" / name, destination / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-image", default="localhost/chemgraph-eval:latest")
    parser.add_argument("--proxy-image", default="localhost/chemgraph-eval-proxy:latest")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="chemgraph-image-") as temporary:
        stage_context(temporary)
        for recipe, tag in [("Worker.Containerfile", args.worker_image), ("Proxy.Containerfile", args.proxy_image)]:
            subprocess.run(["podman", "--remote=false", "build", "--http-proxy=false", "--tag", tag,
                            "--file", str(Path(temporary) / recipe), temporary], check=True)


if __name__ == "__main__":
    main()
