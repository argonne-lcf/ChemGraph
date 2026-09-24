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
with five paths:

- **Argo (Argonne)** — enter your ANL domain username; no API key. Works on
  the lab network or VPN.
- **Your own API key** — OpenAI, Anthropic, Google Gemini, Groq, or
  OpenRouter; the key is applied to the server process environment only.
- **ALCF Inference** — click *Log in with Globus*, sign in, and paste the
  authorization code back; tokens are cached under `~/.chemgraph/` and
  refreshed automatically. A token from ALCF's `inference_auth_token.py`
  helper (or an exported `ALCF_ACCESS_TOKEN`) is picked up automatically.
- **Codex (ChatGPT)** — experimental; reuses the ChatGPT login held by the
  Codex CLI on the machine running Streamlit, exactly like
  `chemgraph run --model codex:<id>` (see
  [Codex subscription](codex_subscription.md)). The card reports whether
  the CLI and the optional `chemgraph[codex]` extra are installed and
  whether a ChatGPT login is active (API-key logins are refused). *Sign in
  with ChatGPT* runs `codex login --device-auth` and shows the sign-in URL
  and one-time code in the page, so the login can be completed from any
  browser even when the server is remote; *Log out* runs `codex logout`.
- **Local (Ollama)** — point at a running OpenAI-compatible server.

The **Configuration → Providers** tab offers the same per-provider cards
afterwards: readiness status, credentials, endpoint settings, and a model
picker with one-click activation.

When started from a source checkout, the app's default configuration path is
the repository-root `config.toml`. For a wheel installation it is
`$XDG_CONFIG_HOME/chemgraph/config.toml` (`~/.config/chemgraph/config.toml` by
default; `%APPDATA%\chemgraph\config.toml` on Windows), because the Python
installation directory is often read-only. Set `CHEMGRAPH_CONFIG` to a file
path to override either choice. The Configuration page shows the active path,
and a failed save is reported instead of being silently dropped. The interface
exposes these workflow choices:

- `single_agent`
- `multi_agent`
- `deep_agent` (experimental; see below)
- `python_relp`
- `graspa`
- `molecular_docking`
- `mock_agent`

Not every CLI workflow is available in Streamlit. Optional workflows still
need their dependencies and external programs.

### Deep Agent

The **Configuration → Deep Agent** tab edits the same `[general]` keys the
CLI reads from `config.toml`, so one file drives both front ends:

| Setting | Key | CLI equivalent |
| --- | --- | --- |
| Workspace directory | `deepagent_workspace` | `--deepagent-workspace` |
| Extra skill directories (one per line) | `deepagent_skills` | `--deepagent-skill` |
| Discover personal and project skills | `deepagent_discover_skills` | `--no-deepagent-discover-skills` |
| Restrict the on-demand tool catalog | `tools` | `--tool` |

The Deep Agent runs shell commands on the host and edits files under the
workspace; the shell is not confined to that directory. Before the agent is
built, the UI asks for the same acknowledgment the CLI's confirmation prompt
does (a checkbox on the Deep Agent tab or in the chat, remembered for the
browser session). Approvals cannot be disabled in the UI.

Shell commands and file mutations pause the run and appear in the chat as
review cards showing the command, the file content, or a diff of the
proposed edit, with **Approve** and **Reject** buttons (several pending
actions get a per-action choice plus *Submit*, *Approve all* and *Reject
all*). Typing a message instead skips the action and returns your text to
the agent as revision instructions, matching the CLI review prompt.

## Optimization steps

When an exchange produces an optimizer trajectory, the **Optimization**
panel links the convergence plot to the geometries: hover or click a step
in the energy / max-force chart to see that structure, drag the slider to
scrub, or press **Play** to animate every step while the marker follows.
Very long trajectories are sampled evenly (first and last step are always
kept). The playback speed is adjustable per exchange.

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
- **Unexpected config:** inspect the file shown under "Configuration file" on
  the Configuration page (repository-root `config.toml` for a source checkout,
  the per-user path for an installed package, or `$CHEMGRAPH_CONFIG`).

See [Troubleshooting](troubleshooting.md) for broader diagnostics.
