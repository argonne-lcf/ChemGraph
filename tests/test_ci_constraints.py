"""Offline checks for CI profile coverage, drift, and constrained installations."""

from itertools import product
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tomllib

import pytest
import yaml

_spec = importlib.util.spec_from_file_location(
    "ci_constraints", Path(__file__).resolve().parents[1] / "scripts/ci_constraints.py")
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)


def test_profiles_cover_ci_jobs_and_metadata():
    config = tomllib.loads((ci.ROOT / "requirements/ci/profiles.toml").read_text())
    jobs = yaml.safe_load((ci.ROOT / ".github/workflows/tests.yml").read_text())["jobs"]
    platforms = {"ubuntu-latest": "linux-x64", "windows-latest": "windows-x64",
                 "macos-latest": "macos-arm64"}
    expected = set()
    for name in ("test", "lint", "test-extras"):
        job = jobs[name]
        matrix = job.get("strategy", {}).get("matrix", {})
        pythons = matrix.get("python-version", ["3.11"])
        for runner, python in product(matrix.get("os", [job["runs-on"]]), pythons):
            expected.add(f"{name}-{platforms[runner]}-py{python}")
        installs = "\n".join(step.get("run", "") for step in job["steps"])
        assert f'pip=={config["pip"]}' in installs
        assert f"path --job {name}" in installs
        assert f"check --job {name} --installed" in installs
        if config["jobs"][name]["extras"]:
            assert f'.[{",".join(config["jobs"][name]["extras"])}]' in installs
    assert set(ci.profiles()) == expected
    for name, spec in ci.profiles().items():
        ci.check_requirements(spec, ci.check_source(name, spec))


@pytest.fixture
def isolated_profile(tmp_path, monkeypatch):
    directory = tmp_path / "requirements/ci"
    directory.mkdir(parents=True)
    for relative in ("pyproject.toml", "requirements/ci/profiles.toml"):
        shutil.copyfile(ci.ROOT / relative, tmp_path / relative)
    monkeypatch.setattr(ci, "ROOT", tmp_path)
    name = "test-macos-arm64-py3.12"
    spec = ci.profiles()[name]
    path = ci.constraint_path(name)
    path.write_text(f"# input-sha256: {ci.fingerprint(spec)}\nexample==1.0\n")
    return name, spec, path


def test_missing_or_stale_profile_cannot_supply_install_path(isolated_profile):
    name, spec, path = isolated_profile
    spec["requirements"].append("new-dependency>=1")
    with pytest.raises(ValueError, match="Stale constraints"):
        ci.check_source(name, spec)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        ci.check_source(name, spec)
    assert not path.exists()


@pytest.mark.parametrize("content", ["example>=1", "example==1.*", "example @ https://example.org/a.whl",
                                     "Example_Name==1\nexample-name==2", "# empty"])
def test_reject_nonexact_or_duplicate_pins(tmp_path, content):
    path = tmp_path / "constraints.txt"
    path.write_text(content)
    with pytest.raises(ValueError):
        ci.read_pins(path)


@pytest.mark.parametrize("field,value", [("system", "Unknown"), ("machine", "armv7")])
def test_unsupported_platform_has_no_fallback(monkeypatch, field, value):
    monkeypatch.setattr(ci.platform, field, lambda: value)
    with pytest.raises(ValueError, match="Unsupported CI profile"):
        ci.current_profile("test", ci.profiles())


def test_wrong_interpreter_rejected_before_reading_file(monkeypatch):
    monkeypatch.setattr(ci, "current_profile", lambda *_: "test-linux-x64-py3.11")
    with pytest.raises(SystemExit) as error:
        ci.main(["path", "--profile", "test-linux-x64-py3.12"])
    assert error.value.code == 1


def test_marker_and_selected_extra_dependency_closure(monkeypatch):
    spec = {**ci.profiles()["test-linux-x64-py3.12"], "requirements": ["root[feature]>=1"]}
    requires = {"root": ['child>=2; extra == "feature"', 'unused; python_version < "3.0"']}
    versions = {"root": "1.0", "child": "2.1"}
    monkeypatch.setattr(ci.metadata, "requires", lambda name: requires.get(name, []))
    monkeypatch.setattr(ci.metadata, "version", versions.__getitem__)
    ci.check_requirements(spec, versions, installed=True)
    with pytest.raises(ValueError, match="Missing or incompatible pin: child"):
        ci.check_requirements(spec, {"root": "1.0"}, installed=True)
    with pytest.raises(ValueError, match="Missing or incompatible pin: root"):
        ci.check_requirements(spec, {"root": "0.5"})
    with pytest.raises(ValueError, match="Installed version differs"):
        ci.check_requirements(spec, {**versions, "child": "2.2"}, installed=True)


def test_target_markers_do_not_use_the_generating_host():
    spec = {**ci.profiles()["test-windows-x64-py3.13"],
            "requirements": ['windows-only==1; sys_platform == "win32"',
                             'linux-only==1; sys_platform == "linux"']}
    ci.check_requirements(spec, {"windows-only": "1"})
    with pytest.raises(ValueError, match="windows-only"):
        ci.check_requirements(spec, {"linux-only": "1"})


def test_generate_preserves_pins_and_failed_resolution(isolated_profile, monkeypatch):
    name, spec, path = isolated_profile
    spec["requirements"] = ["example>=1"]
    before = path.read_bytes()
    monkeypatch.setattr(ci.subprocess, "check_output", lambda *a, **k: "uv 0.8.18\n")

    def fail(command, **kwargs):
        pins = Path(command[command.index("--constraints") + 1]).read_text()
        assert pins == "example==1.0\n"
        assert kwargs["env"]["MACOSX_DEPLOYMENT_TARGET"] == "14.0"
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(ci.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        ci.generate(name, spec)
    assert path.read_bytes() == before

    def refresh(command, **kwargs):
        assert not Path(command[command.index("--constraints") + 1]).read_text()
        Path(command[command.index("--output-file") + 1]).write_text("example==2.0\n")

    monkeypatch.setattr(ci.subprocess, "run", refresh)
    ci.generate(name, spec, upgrades=["example"])
    assert ci.check_source(name, spec) == {"example": "2.0"}


def test_generation_rejects_wrong_resolver_and_missing_seed(isolated_profile, monkeypatch):
    name, spec, path = isolated_profile
    monkeypatch.setattr(ci.subprocess, "check_output", lambda *a, **k: "uv 0.1.0\n")
    with pytest.raises(ValueError, match="Use uv==0.8.18"):
        ci.generate(name, spec)
    monkeypatch.setattr(ci.subprocess, "check_output", lambda *a, **k: "uv 0.8.18\n")
    with pytest.raises(FileNotFoundError):
        ci.generate(name, spec, seed=path.with_name("missing.txt"))
