# Quickstart

This guide verifies the model connection, runs a lookup, performs a small EMT
calculation, and shows where results are stored.

## 1. Install ChemGraph

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install chemgraph
```

Prefer [uv](https://docs.astral.sh/uv/)? Use it to create the same environment
and install ChemGraph:

```bash
uv venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
uv pip install chemgraph
```

See [Installation](installation.md) if you need source or optional-extra setup.

## 2. Choose one model route

For OpenAI:

```bash
export OPENAI_API_KEY="..."
```

For Argonne's Argo gateway:

```bash
export ARGO_USER="<anl-username>"
```

Or start Ollama locally and use `--model llama3.2` without an API key. See
[Models and authentication](models.md) for other providers and exact model IDs.

List registered models and check credentials:

```bash
chemgraph models
chemgraph run --check-keys
```

The commands below use the default `gpt-4o-mini`. For Argo, add
`--model argo:gpt-4o` to each `chemgraph run` command.

## 3. Run a lookup

```bash
chemgraph run -q "What is the SMILES string for aspirin?"
```

This request uses an LLM and PubChem, so it requires network access. Add `-v`
for INFO diagnostics or `-vv` for DEBUG output.

## 4. Run a small calculation

EMT is bundled, fast, and requires no model download:

```bash
chemgraph run \
  --output last_message \
  -q "Build water from SMILES O, optimize it with EMT, and report the final energy."
```

Agent-generated tool selection is probabilistic. Explicitly naming EMT makes
this a more reliable installation check than asking the agent to choose any
calculator.

## 5. Inspect results

Each run creates a session directory under `cg_logs/`, for example:

```text
cg_logs/session_20260101_120000_a1b2c3d4/
```

Depending on the tools called, it can contain structures, trajectories, JSON or
CSV results, spectra, and HTML reports. Direct artifacts elsewhere by setting
the environment variable before starting ChemGraph:

```bash
export CHEMGRAPH_LOG_DIR="$PWD/my_chemgraph_runs"
```

## 6. Continue

- Learn output, session, and interactive options in the [CLI guide](cli.md).
- Compare agent architectures in [Workflows](workflows.md).
- Review model and scientific-engine requirements in [Calculators](calculators.md).
- Use ChemGraph from an application with the [Python API](python_api.md).
- Diagnose setup problems with [Troubleshooting](troubleshooting.md).
