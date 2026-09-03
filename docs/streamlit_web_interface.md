# Streamlit interface

The web interface provides model/workflow selection, chat, molecular
visualization, saved sessions, and access to run artifacts.

## Run from a source checkout

The Streamlit entry point is currently distributed in the repository source
tree rather than as a standalone console script:

```bash
git clone https://github.com/argonne-lcf/ChemGraph.git
cd ChemGraph
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
chemgraph ui
```

Open `http://localhost:8501`. `chemgraph ui` works from any directory
(`--address`, `--port`, and `--headless` are supported); from the repo root
you can also run `streamlit run src/ui/app.py` directly.

## Run with Docker

```bash
docker run --rm -it \
  -e OPENAI_API_KEY \
  -p 8501:8501 \
  ghcr.io/argonne-lcf/chemgraph:latest \
  streamlit run src/ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501
```

Set the credential in the host environment first. See [Docker](docker_support.md)
for artifact volumes and other modes.

## Configure the interface

On first launch (no provider configured) the chat page shows a setup screen
with four paths:

- **Argo (Argonne)** — enter your ANL domain username; no API key. Works on
  the lab network or VPN.
- **Your own API key** — OpenAI, Anthropic, Google Gemini, Groq, or
  OpenRouter; the key is applied to the server process environment only.
- **ALCF Inference** — click *Log in with Globus*, sign in, and paste the
  authorization code back; tokens are cached under `~/.chemgraph/` and
  refreshed automatically. A token from ALCF's `inference_auth_token.py`
  helper (or an exported `ALCF_ACCESS_TOKEN`) is picked up automatically.
- **Local (Ollama)** — point at a running OpenAI-compatible server.

The **Configuration → Providers** tab offers the same per-provider cards
afterwards: readiness status, credentials, endpoint settings, and a model
picker with one-click activation.

When started from the checkout, the app's default configuration path is the
repository-root `config.toml`. The interface exposes these workflow choices:

- `single_agent`
- `multi_agent`
- `python_relp`
- `graspa`
- `molecular_docking`
- `mock_agent`

Not every CLI workflow is available in Streamlit. Optional workflows still
need their dependencies and external programs.

API credentials entered in the UI should be treated as secrets. Prefer
environment variables for shared deployments, avoid placing tokens in a
committed TOML file, and protect network access to the Streamlit server.

## Attachments

Attach structure or data files (XYZ, PDB, CIF, JSON, CSV, ...) with the
paperclip in the chat box and refer to them in your message ("optimize the
attached structure"). Files are saved into the exchange's artifact
directory and the agent receives their exact paths.

## Sessions and artifacts

The UI maintains chat state and exposes prior sessions through its session
controls. Chemistry tools write artifacts to the same session-aware log layout
as the CLI. Set `CHEMGRAPH_LOG_DIR` before starting Streamlit to redirect them.

## Common problems

- **App path not found:** run the command from a source checkout root, or use
  the Docker image.
- **Model absent or unauthorized:** run `chemgraph models` and
  `chemgraph run --check-keys` in the same environment.
- **Calculator missing:** install its optional extra or external executable;
  use EMT for a basic check.
- **Remote browser cannot connect:** bind Streamlit to an appropriate interface
  only on a trusted network and follow your site's port-forwarding policy.
- **Unexpected config:** inspect the repository-root `config.toml` used by the
  source app.

See [Troubleshooting](troubleshooting.md) for broader diagnostics.
