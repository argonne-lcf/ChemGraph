"""Generate and verify the dependency constraints consumed by tests.yml."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def profiles():
    config = tomllib.loads((ROOT / "requirements/ci/profiles.toml").read_text())
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    result = {}
    for job, settings in config["jobs"].items():
        requirements = list(project["dependencies"])
        for extra in settings["extras"]:
            requirements.extend(project["optional-dependencies"][extra])
        requirements.append(f"pip=={config['pip']}")
        if job == "lint":
            requirements.append(f"ruff=={config['ruff']}")
        for name in settings["platforms"]:
            for python in settings["python"]:
                identifier = f"{job}-{name}-py{python}"
                result[identifier] = {
                    "job": job, "python": python, "platform": name,
                    **config["platforms"][name], "requirements": requirements,
                    "requires_python": project["requires-python"],
                    "resolver": config["resolver"],
                }
    return result


def current_profile(job, entries):
    matches = [name for name, spec in entries.items()
               if spec["job"] == job
               and spec["python"] == f"{sys.version_info.major}.{sys.version_info.minor}"
               and spec["system"] == platform.system()
               and spec["machine"].lower() == platform.machine().lower()]
    if len(matches) != 1:
        raise ValueError(f"Unsupported CI profile: {job}, {platform.system()}, "
                         f"{platform.machine()}, Python {platform.python_version()}")
    return matches[0]


def constraint_path(identifier):
    return ROOT / "requirements/ci" / f"{identifier}.txt"


def fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def read_pins(path):
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([A-Za-z0-9.+!_-]+)", line)
        if not match or normalize(match[1]) in pins:
            raise ValueError(f"Invalid or duplicate exact pin in {path.name}: {line}")
        pins[normalize(match[1])] = match[2]
    if not pins:
        raise ValueError(f"No pins in {path.name}")
    return pins


def check_source(identifier, spec):
    path = constraint_path(identifier)
    expected = f"# input-sha256: {fingerprint(spec)}"
    if expected not in path.read_text(encoding="utf-8").splitlines():
        raise ValueError(f"Stale constraints: {identifier}; regenerate this profile")
    return read_pins(path)


def check_requirements(spec, pins, installed=False):
    # The path command needs only the standard library, before installing dependencies.
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet

    environment = default_environment()
    if not installed:
        environment.update(python_version=spec["python"],
                           python_full_version=spec["python"] + ".0",
                           platform_system=spec["system"],
                           platform_machine=spec["machine"],
                           sys_platform=spec["sys_platform"])
    if not SpecifierSet(spec["requires_python"]).contains(environment["python_full_version"]):
        raise ValueError("Profile Python is outside requires-python")
    pending = [(text, ("",)) for text in spec["requirements"]]
    visited = set()
    while pending:
        text, extras = pending.pop()
        requirement = Requirement(text)
        if requirement.marker and not any(
            requirement.marker.evaluate({**environment, "extra": extra}) for extra in extras
        ):
            continue
        name = normalize(requirement.name)
        if name not in pins or not requirement.specifier.contains(pins[name]):
            raise ValueError(f"Missing or incompatible pin: {requirement}")
        key = name, tuple(sorted(requirement.extras))
        if installed and key not in visited:
            visited.add(key)
            pending.extend((child, ("", *requirement.extras))
                           for child in metadata.requires(name) or [])
    if installed:
        for name, version in pins.items():
            if metadata.version(name) != version:
                raise ValueError(f"Installed version differs from pin: {name}=={version}")


def generate(identifier, spec, seed=None, upgrades=()):
    version = subprocess.check_output(["uv", "--version"], text=True).split()[1]
    if version != spec["resolver"]:
        raise ValueError(f"Use uv=={spec['resolver']}, found {version}")
    destination = constraint_path(identifier)
    source = Path(seed) if seed else destination
    pins = read_pins(source) if seed or source.exists() else {}
    for name in upgrades:
        name = normalize(name)
        if name not in pins:
            raise ValueError(f"Cannot refresh an absent pin: {name}")
        del pins[name]
    with tempfile.TemporaryDirectory(prefix="chemgraph-constraints-") as temporary:
        temporary = Path(temporary)
        inputs, constraints, output = [temporary / name for name in ("input.in", "pins.txt", "output.txt")]
        inputs.write_text("\n".join(spec["requirements"]) + "\n", encoding="utf-8")
        constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(pins.items())),
                               encoding="utf-8")
        subprocess.run([
            "uv", "pip", "compile", str(inputs), "--constraints", str(constraints),
            "--python-version", spec["python"], "--python-platform", spec["target"],
            "--default-index", "https://pypi.org/simple", "--no-header", "--no-annotate",
            "--output-file", str(output), "--quiet",
        ], check=True, env={**os.environ, **(
            {"MACOSX_DEPLOYMENT_TARGET": spec["deployment_target"]}
            if "deployment_target" in spec else {}
        )})
        resolved = read_pins(output)
        check_requirements(spec, resolved)
        content = (f"# Generated by uv=={version}; use scripts/ci_constraints.py generate.\n"
                   f"# profile: {identifier}\n# input-sha256: {fingerprint(spec)}\n"
                   + "".join(f"{name}=={pin}\n" for name, pin in sorted(resolved.items())))
        destination.write_text(content, encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("path", "check", "generate"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--profile")
    selection.add_argument("--job", choices=("test", "lint", "test-extras"))
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--seed", help="Initial constraints recovered from a successful CI job")
    parser.add_argument("--upgrade-package", action="append", default=[])
    args = parser.parse_args(argv)
    try:
        entries = profiles()
        identifier = current_profile(args.job, entries) if args.job else args.profile
        if identifier and identifier not in entries:
            raise ValueError(f"Unknown profile: {identifier}")
        if args.command != "check" and not identifier:
            raise ValueError("Select --profile or --job")
        if args.installed and (args.command != "check" or not identifier):
            raise ValueError("--installed requires check and one selected profile")
        if (args.seed or args.upgrade_package) and args.command != "generate":
            raise ValueError("--seed and --upgrade-package require generate")
        if args.command == "generate":
            generate(identifier, entries[identifier], args.seed, args.upgrade_package)
        elif args.command == "path":
            if current_profile(entries[identifier]["job"], entries) != identifier:
                raise ValueError(f"Profile does not match this interpreter: {identifier}")
            check_source(identifier, entries[identifier])
            print(constraint_path(identifier))
        else:
            for name in [identifier] if identifier else entries:
                spec = entries[name]
                if args.installed and current_profile(spec["job"], entries) != name:
                    raise ValueError(f"Profile does not match this interpreter: {name}")
                check_requirements(spec, check_source(name, spec), args.installed)
            print("CI constraints verified.")
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
