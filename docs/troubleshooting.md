# Troubleshooting

Start with the narrowest diagnostic commands:

```bash
python --version
which chemgraph
chemgraph --help
chemgraph models
chemgraph run --check-keys
```

Add `-v` for INFO logs or `-vv` for DEBUG logs to a failing `chemgraph run`.

## Command not found

Activate the environment in which ChemGraph was installed:

```bash
source .venv/bin/activate
python -m pip show chemgraph
```

Python 3.11 or newer is required.

## Credential check fails

- Confirm that the selected model and environment variable belong to the same
  provider.
- Export the variable in the shell that launches ChemGraph, Streamlit, or
  Docker.
- Use `ARGO_USER` for `argo:` routes and `ALCF_ACCESS_TOKEN` for ALCF routes.
- Do not quote placeholders literally or place secrets in committed config.

See [Models and authentication](models.md).

## Calculator is unavailable

ChemGraph only registers engines it can import or locate. Name the calculator
explicitly in your prompt, install its extra, and verify external executables
separately. EMT is the simplest bundled smoke test. MACE can download weights
on first use, and UMA is best installed in a separate environment.

See [Calculators](calculators.md).

## A run appears to hang

The first call may be downloading model or calculator data. Use `-vv` to locate
the active step and inspect the latest directory under `cg_logs/`. Live LLM and
database calls also require network access.

## Results are not where expected

Relative artifact paths are resolved under the current ChemGraph session log
directory. The default parent is `cg_logs/`. Set an absolute location before
launching the process:

```bash
export CHEMGRAPH_LOG_DIR="/absolute/path/to/runs"
```

## A source checkout imports the wrong ChemGraph

A stale site-packages installation can shadow the checkout. Activate a clean
environment and install the current checkout editable:

```bash
python -m pip install -e .
python -c "import chemgraph; print(chemgraph.__file__)"
```

For test isolation, `PYTHONNOUSERSITE=1` can prevent user-site packages from
being imported.

## Streamlit cannot find the app or config

`src/ui/app.py` is a source-tree entry point. Run it from the repository root,
or use Docker. Its default `config.toml` is also rooted at the checkout. See the
[Streamlit guide](streamlit_web_interface.md).

## An MCP client cannot connect

For streamable HTTP, include the path and trailing slash:

```text
http://localhost:9003/mcp/
```

Confirm that server and client use the same transport. Stdio clients launch the
module as a subprocess and do not connect to an HTTP port. See
[MCP servers](mcp_servers.md).

## A saved session cannot be resumed

CLI sessions are stored in `~/.chemgraph/sessions.db`. List them with
`chemgraph session list`, inspect one with `chemgraph session show <id>`, and
resume with the same workflow. The checkpointed `main_agent` must run in
interactive mode.

## Still blocked

Search [GitHub issues](https://github.com/argonne-lcf/ChemGraph/issues). When
opening an issue, include the ChemGraph version, Python version, operating
system, selected workflow/calculator, sanitized command, and relevant DEBUG
output. Never include tokens or proprietary calculation inputs.
