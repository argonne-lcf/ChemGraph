"""Install the specialist OCSR models, one isolated environment per model.

    python -m chemgraph.tools.ocsr_setup --list
    python -m chemgraph.tools.ocsr_setup decimer
    python -m chemgraph.tools.ocsr_setup --check

ChemGraph already accepts that some dependencies cannot share an environment:
`docs/installation.md` tells users to keep UMA and MACE apart because their `e3nn`
requirements conflict. The four OCSR specialists are the same situation, four times
over and worse. They need Python 3.8, 3.10, 3.10 and 3.11, and MolScribe's torch 1.13
is compiled against numpy 1.x while ChemGraph pins numpy 2.2.6. There is no single
environment that holds all of them, so each gets its own and is driven as a
subprocess.

This module automates what the UMA note leaves to the reader. It does not invent a
packaging mechanism: it runs the build script for one model, which creates a conda
env, pip-installs pinned versions, and fetches weights from HuggingFace. Everything
it needs ships with ChemGraph; nothing is fetched from a private repository.

Nothing here runs automatically. Installing all four is a decision the user makes
explicitly, so this is a command they type, never something a tool call triggers.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
from importlib import resources

# What to install and where, read from the same registry that defines the backends.
# A specialist added to ocsr_registry.json is installable here with no edit, and there
# is no second list to fall out of step with the first.
def _install_specs() -> dict:
    from chemgraph.tools.ocsr_models import load_registry

    out = {}
    for name, m in load_registry()["specialists"].items():
        install = dict(m.get("install") or {})
        if not install.get("script"):
            continue  # a model with no build script is not installable by this tool
        install["note"] = m.get("install_note", m.get("note", ""))
        out[name] = install
    return out


MODELS = _install_specs()

_INSTALL_JSON = "~/.chemgraph/ocsr_install.json"
_INSTALL_ENV = "CHEMGRAPH_OCSR_INSTALL"


def load_install(path: str | None = None) -> dict:
    """Where each model is installed: explicit path, env var, user file, then defaults.

    The user file is what :func:`_record_install` writes, so a model installed through
    ``ocsr_setup`` is found with no further configuration. The packaged fallback is
    derived from :data:`MODELS` rather than shipped as JSON, since every path in it is
    per-user and a wheel cannot know them.
    """
    candidate = path or os.environ.get(_INSTALL_ENV) or _INSTALL_JSON
    with contextlib.suppress(FileNotFoundError, json.JSONDecodeError):
        with open(os.path.expanduser(candidate)) as fh:
            return json.load(fh)
    return {
        name: {
            "python_bin": _env_python(name),
            "weights_path": os.path.expanduser(m["weights"]),
            "device": "cpu",
            "startup_timeout_s": m.get("startup_timeout_s", 300),
            "timeout_s": m.get("timeout_s", 120),
        }
        for name, m in MODELS.items()
    }


def _script_path(name: str) -> str:
    """Absolute path to a build script, extracting it from the wheel if needed.

    Raises FileNotFoundError if the registry names a script that is not there.
    ``importlib.resources`` only composes a path, so without this check a typo'd
    registry entry returns a plausible-looking path and fails later as an opaque
    "bash: no such file", after the caller has already printed a download size and
    asked for confirmation.
    """
    script = MODELS[name]["script"]
    ref = resources.files("chemgraph.tools.ocsr_workers").joinpath(script)
    if not ref.is_file():
        raise FileNotFoundError(
            f"the registry entry for {name!r} names build script {script!r}, but "
            f"chemgraph/tools/ocsr_workers/{script} does not exist. Add the script, "
            f"or correct 'install.script' in ocsr_registry.json."
        )
    with resources.as_file(ref) as p:
        return str(p)


def _env_python(name: str) -> str:
    return os.path.expanduser(os.path.join(MODELS[name]["env"], "bin", "python"))


def cmd_list() -> int:
    """Print what is available, what it costs, and what is already installed.

    Accuracy comes from the calibration table rather than this module, so the listing
    reports whatever table is in force. A model with no measured accuracy prints a
    dash: unmeasured is not the same as zero.
    """
    from chemgraph.tools.ocsr_core import load_calibration
    from chemgraph.tools.ocsr_models import DEFAULT_SPECIALIST

    try:
        performance = load_calibration().get("model_performance") or {}
    except (OSError, ValueError, TypeError):
        performance = {}

    print("Specialist OCSR models. Each installs into its own conda environment.\n")
    print(f"  {'model':11s} {'python':7s} {'disk':>7s} {'accuracy':>9s}  {'status':12s}")
    for name, m in MODELS.items():
        total = m["size_gb"] + m["weights_gb"]
        status = "installed" if os.path.exists(_env_python(name)) else "not installed"
        accuracy = performance.get(name, {}).get("accuracy")
        shown = f"{accuracy:8.1%}" if accuracy is not None else f"{'--':>8s}"
        print(f"  {name:11s} {m['python']:7s} {total:6.1f}G {shown}  {status:12s}")
    total_all = sum(m["size_gb"] + m["weights_gb"] for m in MODELS.values())
    print(f"\n  {DEFAULT_SPECIALIST} alone is enough for backend='auto'.")
    print(f"  All {len(MODELS)} ({total_all:.1f} GB) are only needed for "
          f"backend='ensemble'.")
    print("\n  Install one with:  python -m chemgraph.tools.ocsr_setup <model>")
    return 0


def _missing_registry_files(name: str) -> list[str]:
    """Files the registry promises for a model but that are not in the package.

    Distinguishes a broken registry entry from an uninstalled model. Since the
    registry is user-editable data, a typo in a filename is an expected mistake, not
    a developer bug -- and without this it surfaces 45 seconds into a build as
    "bash: no such file", or worse, as a model that resolves as a backend and then
    fails at first call.
    """
    from chemgraph.tools.ocsr_models import load_registry

    spec = load_registry()["specialists"].get(name) or {}
    workers = resources.files("chemgraph.tools.ocsr_workers")
    missing = []
    for key in ("worker", ("install", "script")):
        if isinstance(key, tuple):
            filename = (spec.get(key[0]) or {}).get(key[1])
        else:
            filename = spec.get(key)
        if filename and not workers.joinpath(filename).is_file():
            missing.append(filename)
    return missing


def cmd_check() -> int:
    """Verify what is actually usable, and say what to run for anything missing."""
    ok = True
    broken_registry = False
    print(f"  {'model':11s} {'env':38s} {'status'}")
    for name in MODELS:
        absent = _missing_registry_files(name)
        if absent:
            # Report this before the env check: telling the user to install a model
            # whose build script does not exist sends them into a 45 s failure.
            print(f"  {name:11s} {MODELS[name]['env']:38s} "
                  f"REGISTRY ERROR: no such file: {', '.join(absent)}")
            ok = False
            broken_registry = True
            continue
        py = _env_python(name)
        if not os.path.exists(py):
            print(f"  {name:11s} {MODELS[name]['env']:38s} not installed")
            ok = False
            continue
        # The interpreter existing is not the same as it working: a user-site package
        # can shadow the env and break an import that only fails at model load.
        probe = subprocess.run(
            [py, "-c", "import sys; print(sys.version_info[:2])"],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        if probe.returncode != 0:
            print(f"  {name:11s} {MODELS[name]['env']:38s} BROKEN ({probe.stderr.strip()[:40]})")
            ok = False
        else:
            print(f"  {name:11s} {MODELS[name]['env']:38s} ok  python {probe.stdout.strip()}")
    if broken_registry:
        print("\n  A REGISTRY ERROR is not something installing will fix: the entry in")
        print("  ocsr_registry.json names a file that is not in chemgraph/tools/")
        print("  ocsr_workers/. Add the file, or correct the name in the registry.")
    if ok:
        return 0
    if not broken_registry:
        print("\n  Install a missing model with:")
        print("    python -m chemgraph.tools.ocsr_setup <model>")
    return 1


def cmd_install(name: str, yes: bool = False) -> int:
    """Run one model's build script, after saying what it will cost."""
    m = MODELS[name]
    script = _script_path(name)
    total = m["size_gb"] + m["weights_gb"]

    print(f"Installing {name}")
    print(f"  environment  {m['env']}  (conda, python {m['python']})")
    print(f"  weights      ~{m['weights_gb']:.1f} GB from HuggingFace")
    print(f"  total disk   ~{total:.1f} GB")
    print(f"  script       {script}")
    print(f"\n  {m['note']}")

    if not shutil.which("conda"):
        print("\nconda not found. Install miniforge, or load your site's conda module:")
        print("  https://github.com/conda-forge/miniforge")
        return 1

    if not yes:
        try:
            if input("\nProceed? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Nothing was installed.")
                return 1
        except EOFError:
            print("\nNot a terminal; re-run with --yes to install non-interactively.")
            return 1

    print()
    rc = subprocess.call(["bash", script])
    if rc != 0:
        print(f"\nBuild failed (exit {rc}). The script's output above says why.")
        return rc

    _record_install(name)
    print(f"\n{name} installed. Verify with:")
    print("  python -m chemgraph.tools.ocsr_setup --check")
    return 0


def _record_install(name: str) -> None:
    """Note where this model landed, so the tool finds it without a hardcoded path."""
    path = os.path.expanduser(_INSTALL_JSON)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    # Everything a worker needs to start. Three of the four declare --weights as
    # required, so recording only python_bin would produce a config that dies at
    # argparse 0.3 s into the spawn.
    m = MODELS[name]
    data[name] = {
        "python_bin": _env_python(name),
        "weights_path": os.path.expanduser(m["weights"]),
        "device": "cpu",
        "startup_timeout_s": m.get("startup_timeout_s", 300),
        "timeout_s": m.get("timeout_s", 120),
    }
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)


def build_parser() -> argparse.ArgumentParser:
    """The CLI's argument parser.

    Separate from main() so a test can check that the command this package tells
    users to run is a command this package accepts.
    """
    p = argparse.ArgumentParser(
        prog="python -m chemgraph.tools.ocsr_setup",
        description="Install specialist OCSR models into isolated environments.",
    )
    p.add_argument("model", nargs="?", choices=sorted(MODELS), help="model to install")
    p.add_argument("--list", action="store_true", help="show models, sizes and status")
    p.add_argument("--check", action="store_true", help="verify installed environments")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.check:
        return cmd_check()
    if args.list or not args.model:
        return cmd_list()
    return cmd_install(args.model, yes=args.yes)


if __name__ == "__main__":
    sys.exit(main())
