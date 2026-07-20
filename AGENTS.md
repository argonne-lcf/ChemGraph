# AGENTS.md

Guidance for AI coding agents (Claude Code, Cursor, Copilot, etc.) working in
this repository. Human contributors should read
[CONTRIBUTING.md](CONTRIBUTING.md); this file is the machine-facing companion.

## Project overview

ChemGraph is a computational-chemistry agent framework (PyPI package
`chemgraph`, import package `chemgraph`). It connects natural-language queries to
molecular simulations via a LangGraph/LangChain agent architecture, ASE, RDKit,
and MCP servers. Requires Python >= 3.10.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .                                  # core
pip install -e ".[academy,parsl,globus_compute]"  # to run the HPC/Academy tests
```

Optional extras: `calculators`, `uma`, `ui`, `parsl`, `ensemble_launcher`,
`globus_compute`, `academy`, `xanes`, `rag`.

## Build / test / lint commands

```bash
# Lint (must pass — CI gate)
ruff check .

# Core test suite (CI gate)
pytest tests/ -k "not tblite"

# Academy + execution-backend tests (need the extras installed)
pytest tests/ -k "not tblite"      # after installing [academy,parsl,globus_compute]

# Run a single test file / test
pytest tests/test_graphs.py -q
pytest tests/test_graphs.py::test_name -q
```

- Tests that call live LLMs are gated behind `--run-llm`; tests needing a live
  Globus Compute endpoint behind `--run-globus-compute`. **Do not** enable these
  in normal runs — they need credentials/endpoints and won't work in CI.
- `tblite` is excluded via `-k "not tblite"` because it isn't always installable.
- Ruff config lives in `pyproject.toml` (`[tool.ruff]`); see
  `docs/code_formatting_and_linting.md`.

## Repository layout

```
src/chemgraph/
  agent/        # ChemGraph agent, run_turn, events
  graphs/       # LangGraph single-/multi-agent graph construction
  tools/        # chemistry + ASE tools (many file readers/writers)
  mcp/          # MCP servers (incl. *_mcp_hpc.py) + cg_fastmcp framework
  execution/    # pluggable backends: local, parsl, globus_compute, job_tracker
  academy/      # distributed multi-agent module (campaigns, dashboard, runtime)
  models/       # LLM loader / settings / provider normalization
  schemas/      # pydantic schemas (ase_input, calculators, ...)
  prompt/ eval/ hpc_configs/ cli/ state/ utils/ memory/
tests/          # pytest suite
docs/           # mkdocs site
```

## Conventions

- **Match the surrounding code** — naming, comment density, and idioms. Don't
  introduce a new style or reformat unrelated code.
- **File paths:** tools that write artifacts resolve relative paths against
  `CHEMGRAPH_LOG_DIR` via `ase_core._resolve_path`; readers resolve via
  `ase_core._resolve_existing_path`. Reuse these helpers rather than re-deriving
  path logic.
- **Optional dependencies** must be imported lazily (inside functions or behind
  `try/except ImportError`) so core install and test collection never break when
  an extra is absent. Follow the pattern in `execution/` and `mcp/`.
- Add or update **tests** for any behavior you change; keep them hermetic (no
  network, no model downloads — e.g. use the EMT calculator).
- The package **version** is single-sourced in `pyproject.toml`.

## Workflow rules (same as humans)

- Branch off the latest `main`; there is **no `dev` branch**. Name branches
  `feature/…`, `fix/…`, `docs/…`, `chore/…`.
- Keep changes **small and focused** — one logical change per PR (~≤400 lines
  where practical). Split large work into incremental PRs.
- Before proposing a change, ensure `ruff check .` and
  `pytest tests/ -k "not tblite"` pass.
- **Never** commit secrets (API keys, PyPI tokens) or paste them into prompts.
- Don't push directly to `main`; open a PR. Don't force-push shared branches.

## Gotchas

- The core dependency `pyppeteer` pins `websockets<11`; MCP HTTP servers run
  uvicorn with `ws="none"` to avoid the resulting protocol import error. Keep
  that when touching `mcp/server_utils.py`.
- A stale `chemgraph`/`chemgraphagent` install in site-packages can shadow the
  editable checkout; run tests in a venv with `-e .` installed (or
  `PYTHONNOUSERSITE=1`) to be sure you're testing the working tree.
