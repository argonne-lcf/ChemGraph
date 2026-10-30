# Reproducing CI dependencies

These constraints describe the tested installations in `tests.yml`, not the
dependencies imposed on downstream users. `pyproject.toml` remains authoritative.
The profiles cover core tests on Linux x64, Windows x64, and macOS ARM64 with
Python 3.11–3.13; lint on Linux/Python 3.11; and the test, Academy, Parsl,
Globus Compute, and Codex extras on Linux/Python 3.11–3.12. macOS resolution
targets macOS 14 or later, matching the Torch wheels used by CI.

In a fresh virtual environment on a supported platform, reproduce the core job:

```bash
python -m pip install "pip==26.2.2"
constraints="$(python scripts/ci_constraints.py path --job test)"
python -m pip install -c "$constraints" -e .
python -m pip check
python scripts/ci_constraints.py check --job test --installed
python -m pytest tests/ -k "not tblite"
```

The example uses Bash (also available on Windows CI). In PowerShell, use
`$constraints = python scripts/ci_constraints.py path --job test` instead.
For the extras job, select `--job test-extras` and install
`-e ".[test,academy,parsl,globus_compute,codex]"`. For lint, select `--job lint`
and install `-e . ruff`. Unsupported platforms, Python versions, and missing or
stale files fail explicitly; there is no unconstrained fallback.

## Updating constraints

Install `uv==0.8.18` in a separate maintenance environment; the checker needs
`packaging`, which is already present in a ChemGraph development environment.
`profiles.toml` defines the job/platform combinations and tool versions.
The initial seeds came from the successful CI run linked there. Regeneration
reads project dependencies directly and preserves every existing pin by default:

```bash
python scripts/ci_constraints.py generate --profile test-macos-arm64-py3.12
python scripts/ci_constraints.py check
```

To update packages deliberately, repeat `--upgrade-package NAME` for each pin
allowed to change. If an update requires another pinned dependency to change,
resolution fails until that dependency is explicitly included. Refresh each
affected profile, review the version diff (especially scientific packages), and
validate fresh installs on the matching CI runners. CI installs committed
constraints; it never regenerates them. A missing profile can be generated with
an initial `--seed PATH` containing exact pins from a known successful install.

The input digest includes the selected dependency declarations, target and tool
versions. The checker validates direct requirements for every profile; after
installation it also checks the active transitive dependency graph, including
extras, and all installed pins. Changing a dependency declaration requires
regenerating affected files even if the resolved versions remain unchanged.

These files reproduce Python package versions, not operating-system images or
byte-identical builds. Build-isolation dependencies and external binaries are
not locked. Wheel-install, `dependency_tests.yml` (including its UMA workaround),
Conda and container jobs retain their existing installation policies. No model
downloads or live integration flags are needed for verification.
