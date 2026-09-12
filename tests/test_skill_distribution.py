"""Build distributions and read skills without an editable source checkout."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


@pytest.fixture(scope="module")
def skill_distributions(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    build_root = tmp_path_factory.mktemp("skill-distribution")
    shutil.copytree(
        root / "src",
        build_root / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
    )
    for name in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy(root / name, build_root / name)
    distribution = build_root / "dist"
    distribution.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_sdist, build_wheel; build_sdist('dist'); build_wheel('dist')",
        ],
        cwd=build_root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return root, distribution


def test_skill_resources_in_wheel_and_sdist(skill_distributions):
    root, distribution = skill_distributions
    expected = {
        path.relative_to(root / "src").as_posix()
        for path in (root / "src/chemgraph/skills").rglob("*")
        if path.is_file() and path.suffix not in {".py", ".pyc"}
    }
    with zipfile.ZipFile(next(distribution.glob("*.whl"))) as wheel:
        assert expected <= set(wheel.namelist())
    with tarfile.open(next(distribution.glob("*.tar.gz"))) as sdist:
        members = {
            name.split("/src/", 1)[1] for name in sdist.getnames() if "/src/" in name
        }
        assert expected <= members


@pytest.mark.parametrize("zip_import", [False, True])
def test_installed_skills_readable_outside_checkout(
    skill_distributions, tmp_path, zip_import
):
    _, distribution = skill_distributions
    wheel = next(distribution.glob("*.whl"))
    installed = tmp_path / "installed"
    if zip_import:
        installed = wheel
    else:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(installed)
    environment = dict(os.environ)
    # Dependencies may use a temporary overlay, but remove checkout entries.
    environment["PYTHONPATH"] = os.pathsep.join(
        entry
        for entry in environment.get("PYTHONPATH", "").split(os.pathsep)
        if entry and not (Path(entry) / "chemgraph").exists()
    )
    script = """
import sys
sys.path.insert(0, sys.argv[1])
import chemgraph
assert chemgraph.__file__.startswith(sys.argv[1]), chemgraph.__file__
from chemgraph.skills.backend import BundledSkillsBackend
backend = BundledSkillsBackend()
assert backend.read('/pbs-hpc/SKILL.md').error is None
assert b'PBS' in backend.download_files(['/pbs-hpc/assets/job.pbs.template'])[0].content
assert backend.write('/pbs-hpc/SKILL.md', 'overwrite').error
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(installed)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
