# AGENTS.md

Guidance for AI coding agents working in this repository. Human contributors
should read [CONTRIBUTING.md](CONTRIBUTING.md); this file is the machine-facing
companion.

## Project overview

ChemGraph is a computational-chemistry agent framework (PyPI package
`chemgraph`, import package `chemgraph`). It connects natural-language queries to
molecular simulations via a LangGraph/LangChain agent architecture, ASE, RDKit,
and MCP servers. Requires Python >= 3.11.

## Setup

Requires Python >= 3.11 (prefer `uv` over `conda` for local virtualenvs).

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .                                        # core
uv pip install -e ".[test,academy,parsl,globus_compute]"   # full CI test extras
```

Optional extras in `pyproject.toml`: `test`, `calculators`, `uma`, `ui`,
`parsl`, `ensemble_launcher`, `globus_compute`, `academy`, `xanes`, `rag`,
`ocsr`, `codex`, `docking`.

## Build / test / lint commands

```bash
# Lint (required CI gate)
ruff check .

# Core test suite (required CI gate)
pytest tests/ -k "not tblite"

# Run a single test file or test
pytest tests/test_graphs.py -q
pytest tests/test_graphs.py::test_name -q

# Lint bundled Agent Skills (required when touching src/chemgraph/skills/)
chemgraph skills lint src/chemgraph/skills --core

# Verify package metadata / dependency consistency
uv pip check
```

- Live integration tests are gated behind `--run-llm`, `--run-globus-compute`,
  and `--run-ensemble-launcher`. **Do not** pass these flags in normal runs —
  they require external credentials/endpoints.
- `CHEMGRAPH_TEST_MACE=1` enables MACE integration tests (skipped or mocked by
  default when MACE models/add-ons are absent).
- `tblite` is excluded via `-k "not tblite"` because it is not always
  installable across platforms.
- UI tests (`tests/test_ui_*.py`) automatically block the live `openai-codex`
  SDK via an autouse fixture in `tests/conftest.py`; monkeypatch
  `chemgraph.models.codex._load_codex_sdk` explicitly when testing SDK paths.

## Repository layout

Two top-level packages live under `src/`:

```
src/
  chemgraph/
    agent/        # ChemGraph orchestrator (llm_agent.py), run_turn, events
    graphs/       # LangGraph workflows (single_agent, multi_agent, deep_agent, ...)
    tools/        # Core simulation logic (ase_core.py) + LangChain @tool wrappers
    mcp/          # FastMCP servers (incl. *_mcp_hpc.py, *_mcp_parsl.py) + cg_fastmcp
    execution/    # Pluggable backends: local, parsl, globus_compute, job_tracker
    academy/      # Distributed multi-agent module (campaigns, dashboard, runtime)
    models/       # LLM loader, settings, provider normalization, codex adapter
    registry/     # ToolRegistry and AgentRegistry specs and availability checks
    skills/       # Bundled Agent Skills (*/SKILL.md) shipped as package data
    schemas/      # Pydantic schemas (AtomsData, ASEInputSchema, calculators/*)
    cli/ eval/ hpc_configs/ memory/ prompt/ state/ utils/
  ui/             # Streamlit web app (src/ui/app.py) and UI components
tests/            # Pytest suite
requirements/     # Git-URL add-ons (mace-polar.txt, ocsr-models.txt) disallowed on PyPI
```

## Architecture & conventions

- **Match the surrounding code** — naming, comment density, and idioms. Do not
  reformat unrelated code.
- **Simulation logic vs. tool wrappers:** plain Python simulation functions live
  in `src/chemgraph/tools/ase_core.py` (no `@tool`, `@mcp.tool`, or
  `@python_app` decorators). LangChain tools (`ase_tools.py`), MCP servers
  (`mcp/`), and HPC runners delegate to `ase_core.py`.
- **Artifact file paths:** tools that write files must resolve relative paths
  against `CHEMGRAPH_LOG_DIR` via `ase_core._resolve_path`; tools that read
  files written by sibling tools must use `ase_core._resolve_existing_path`.
- **Optional dependencies** must be imported lazily (inside functions or behind
  `try/except ImportError`) so core installs and pytest collection succeed when
  extras are absent. Test modules requiring optional extras should guard with
  `pytest.importorskip(...)` at module top.
- **Dependency synchronization (`tests/test_package_metadata.py`):**
  - Exact `==` version pins in `pyproject.toml` `[project].dependencies` must
    stay synchronized with `environment.yml`.
  - The `tblite` version pin in `[project.optional-dependencies].calculators`
    must match `environment.yml`, `Dockerfile`, and `Dockerfile.arm`.
  - PyPI rejects direct URL dependencies (`@ git+https://...`), including in
    extras. Keep Git-only packages in `requirements/*.txt`.
  - Python version classifiers in `pyproject.toml` must match the CI test matrix
    in `.github/workflows/tests.yml`.
- **Bundled skills (`src/chemgraph/skills/`):** must include `license`
  (`Apache-2.0`), `metadata.authors`, and `metadata.maintainers` (string keys
  and values) in `SKILL.md` frontmatter, and pass
  `chemgraph skills lint src/chemgraph/skills --core`.
- **Hermetic tests:** keep tests offline (no network, no model downloads — use
  the `EMT` calculator instead of downloading MACE weights).
- The package **version** is single-sourced in `pyproject.toml`.

## Workflow rules

- Branch off `main`; there is **no `dev` branch**. Name branches `feature/…`,
  `fix/…`, `docs/…`, `chore/…`.
- Keep changes **small and focused** — one logical change per PR (~≤400 lines
  where practical).
- Before proposing a change, ensure `ruff check .` and
  `pytest tests/ -k "not tblite"` pass.
- **Never** commit secrets (API keys, PyPI tokens) or paste them into prompts.
- Don't push directly to `main`; open a PR. Don't force-push shared branches.

## Gotchas

- **MCP stdio stdout corruption:** under `stdio` transport, stdout is the
  JSON-RPC channel. All logging must go to `stderr`, and `server_utils.py` sets
  `CHEMGRAPH_LOCAL_SILENCE_STDOUT=1` by default so noisy library prints (e.g.,
  MACE notices) do not corrupt the protocol.
- **MCP streamable-HTTP WebSockets:** `mcp/server_utils.py` runs uvicorn with
  `ws="none"` because streamable-HTTP does not use WebSockets. Preserve this.
- **UMA vs. MACE conflict:** `fairchem-core` (`uma` extra) and `mace-torch`
  have incompatible `e3nn` requirements and cannot be installed in the same
  environment simultaneously (see `.github/workflows/dependency_tests.yml`).
- **Editable install shadowing:** a stale `chemgraph` or `chemgraphagent`
  install in site-packages can shadow the working tree; always run tests in an
  editable install (`uv pip install -e .`) or with `PYTHONNOUSERSITE=1`.

