!!! note
    ChemGraph includes a Streamlit web UI for chat-driven chemistry workflows, structure visualization, and report viewing.

## Run the app

Set provider keys as needed:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
```

Launch:

```bash
streamlit run src/ui/app.py
```

Then open `http://localhost:8501`.

## Features

- Chat interface for single-agent and multi-agent workflows
- Model selection across supported providers
- 3D molecular visualization with `stmol`/`py3Dmol`
- Embedded report display and structure export
- Config editor for `config.toml`

## Troubleshooting

- If 3D rendering is unavailable, install `stmol`:
  `pip install stmol`
- If model calls fail, verify API keys and endpoint settings in `config.toml`.
- If Argo is used, ensure `api.openai.base_url` and optional `api.openai.argo_user` are configured.
