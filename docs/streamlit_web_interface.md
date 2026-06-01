!!! note
    ChemGraph includes a Streamlit web UI for chat-driven chemistry workflows, live tool progress, structure visualization, report viewing, and saved-session management.

## Run the app

Install ChemGraph, then set the provider credentials required by the model you plan to use:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export GROQ_API_KEY="..."
# ALCF inference endpoints:
export ALCF_ACCESS_TOKEN="..."
```

Launch:

```bash
streamlit run src/ui/app.py
```

Then open `http://localhost:8501`.

## Features

- Chat input for single-agent, multi-agent, Python REPL, gRASPA, and mock-agent workflows exposed in the UI.
- Automatic agent initialization from the active `config.toml`.
- Sidebar calculator availability panel showing calculators detected at startup and the selected default.
- Live tool-call status while workflows run.
- Optional human-supervised pauses through the `ask_human` tool.
- 3D molecular visualization with `stmol` and `py3Dmol`, with table/XYZ fallback when the viewer is unavailable.
- Math-aware assistant rendering for LaTeX-style equations, reaction arrows, and thermochemistry expressions.
- Embedded and downloadable HTML reports, IR spectrum artifacts, normal-mode trajectory controls, and structure export.
- Session browser backed by `~/.chemgraph/sessions.db`.
- Configuration editor for `config.toml` plus session-only API key entry.

## Configuration

The UI reads `config.toml` from the working directory where Streamlit is launched. If the file is missing, the app creates one with defaults.

Use the Configuration page for persistent settings:

- `general.model`: default model.
- `general.workflow`: workflow type. The UI accepts `single_agent`, `multi_agent`, `python_relp`, `graspa`, and `mock_agent`; `python_repl` is accepted as an alias for `python_relp`.
- `general.thread`: default LangGraph thread ID.
- `general.recursion_limit`: workflow recursion limit.
- `general.report`: generate HTML reports when supported.
- `general.human_supervised`: allow the agent to pause and request human input.
- `api.*.base_url` and `api.*.timeout`: provider endpoint settings.
- `api.openai.argo_user`: optional Argo username; `ARGO_USER` is used only as a fallback.

API keys entered in the UI are applied as process environment variables for the current Streamlit process and are not saved to `config.toml`. For shared deployments, prefer server-side environment variables.

## Sessions

The main sidebar lists recent saved sessions. Loading a session rebuilds the visible conversation history from `~/.chemgraph/sessions.db`; deleting a session removes it from that database. A new chat clears the visible conversation and starts a new saved session on the next successful exchange.

The UI uses the active saved configuration for model, workflow, thread, report generation, and human-supervision settings. To change these settings, use the Configuration page, save the configuration, then click **Reload Config** or **Refresh Agents** on the main page.

## Artifacts

The UI detects structures and reports from agent messages. For IR calculations, it looks in the run directory referenced by the result message for files such as `ir_spectrum_<name>.png`, `frequencies_<name>.csv`, and `<name>_vib.<mode>.traj`.

## Troubleshooting

- If 3D rendering is unavailable, install `stmol`:
  `pip install stmol`
- If model calls fail, verify API keys and endpoint settings in `config.toml`.
- If Argo is used, ensure `api.openai.base_url` and optional `api.openai.argo_user` are configured.
- If a local model endpoint is selected, the UI probes `/models` and blocks queries when the local endpoint is unreachable.
- If the UI still shows an old model, workflow, or calculator default after editing configuration, click **Reload Config** or **Refresh Agents**.
