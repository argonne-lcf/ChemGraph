!!! note
    Docker setup is now intentionally simplified and does **not** include vLLM.
    The same ChemGraph image can be launched in four modes: JupyterLab, Streamlit UI, MCP server, or interactive CLI.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Files

- `Dockerfile`: standard ChemGraph container image
- `Dockerfile.arm`: ARM-friendly variant (same runtime goals, no vLLM)
- `docker-compose.yml`: profile-based launcher for Jupyter/Streamlit/MCP

## Use Published GHCR Image (No Local Build)

If you do not want a local install, run the published container image directly:

Run JupyterLab:

```bash
docker run --rm -it -p 8888:8888 ghcr.io/argonne-lcf/chemgraph:latest \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --LabApp.token=
```

Run Streamlit:

```bash
docker run --rm -it -p 8501:8501 ghcr.io/argonne-lcf/chemgraph:latest \
  streamlit run src/ui/app.py --server.address=0.0.0.0 --server.port=8501
```

Run MCP server (HTTP):

```bash
docker run --rm -it -p 9003:9003 ghcr.io/argonne-lcf/chemgraph:latest \
  python -m chemgraph.mcp.mcp_tools --transport streamable_http --host 0.0.0.0 --port 9003
```

Run interactive CLI shell:

```bash
docker run --rm -it --entrypoint /bin/bash -v "$PWD:/work" -w /work \
  ghcr.io/argonne-lcf/chemgraph:latest
```

Then inside the container:

```bash
chemgraph --help
chemgraph --config config.toml -q "calculate the energy for smiles=O using mace_mp"
```

## Build Image

From project root:

```bash
docker compose build
```

If you previously built the image before this update, rebuild so the Git safe-directory
setting for `/app` is included.

## Run JupyterLab

```bash
docker compose --profile jupyter up
```

Access: `http://localhost:8888`

## Run Streamlit App

```bash
docker compose --profile streamlit up
```

Access: `http://localhost:8501`

## Run MCP Server (HTTP transport)

```bash
docker compose --profile mcp up
```

MCP endpoint: `http://localhost:9003`

The compose service launches:

```bash
python -m chemgraph.mcp.mcp_tools --transport streamable_http --host 0.0.0.0 --port 9003
```

## Environment Variables

The compose file forwards these variables into the container when set on host:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `CHEMGRAPH_LOG_DIR` (default in compose: `/app/cg_logs`)
- `PYTHONPATH` is set to `/app/src` in compose so bind-mounted source code is used.

Example:

```bash
export OPENAI_API_KEY="your_key"
docker compose --profile streamlit up
```

## Stop Services

```bash
docker compose down
```

## Run Without Compose (Optional)

Build once:

```bash
docker build -t chemgraph:local .
```

Jupyter:

```bash
docker run --rm -it -p 8888:8888 -v "$PWD:/app" chemgraph:local \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --LabApp.token=
```

Streamlit:

```bash
docker run --rm -it -p 8501:8501 -v "$PWD:/app" chemgraph:local \
  streamlit run src/ui/app.py --server.address=0.0.0.0 --server.port=8501
```

MCP (HTTP):

```bash
docker run --rm -it -p 9003:9003 -v "$PWD:/app" chemgraph:local \
  python -m chemgraph.mcp.mcp_tools --transport streamable_http --host 0.0.0.0 --port 9003
```

## Notes

- This setup avoids local model-serving dependencies and keeps Docker usage focused on ChemGraph tooling.
- If you need local LLM serving, run it as a separate service outside this Docker setup and point ChemGraph to that endpoint via model/base URL configuration.
- Compose startup also runs `git config --global --add safe.directory /app` to avoid
  Git "dubious ownership" errors in notebooks/Streamlit when the repo is bind-mounted.
- The default `Dockerfile` installs `nwchem` and `tblite` with conda-forge.

## Publish to GHCR

A GitHub Actions workflow (`.github/workflows/ghcr-publish.yml`) publishes the Docker image to:

- `ghcr.io/<org-or-user>/chemgraph:<tag>`
- `ghcr.io/<org-or-user>/chemgraph:sha-<commit>`

How to publish:

```bash
git tag v0.3.0
git push origin v0.3.0
```

You can also trigger the workflow manually from Actions with `workflow_dispatch`.
