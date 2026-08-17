# Docker

The ChemGraph image can run JupyterLab, Streamlit, the CLI, or the general MCP
server. It includes the source-tree entry points plus NWChem and TBLite support
configured by the repository Dockerfile.

## Streamlit

Set the credential in your host environment, then pass it by name:

```bash
export OPENAI_API_KEY="..."
docker run --rm -it \
  -e OPENAI_API_KEY \
  -p 8501:8501 \
  -v "$PWD/cg_logs:/app/cg_logs" \
  ghcr.io/argonne-lcf/chemgraph:latest \
  streamlit run src/ui/app.py \
    --server.address=0.0.0.0 \
    --server.port=8501
```

Open `http://localhost:8501`. The volume preserves artifacts after exit.

## CLI

```bash
docker run --rm -it \
  -e OPENAI_API_KEY \
  -v "$PWD/cg_logs:/app/cg_logs" \
  ghcr.io/argonne-lcf/chemgraph:latest \
  chemgraph run -q "What is the SMILES string for aspirin?"
```

Forward only the credential needed by the selected provider.

## MCP over HTTP

```bash
docker run --rm -it \
  -p 9003:9003 \
  -v "$PWD/cg_logs:/app/cg_logs" \
  ghcr.io/argonne-lcf/chemgraph:latest \
  python -m chemgraph.mcp.mcp_tools \
    --transport streamable_http \
    --host 0.0.0.0 \
    --port 9003
```

Clients connect to `http://localhost:9003/mcp/`. Limit network exposure; this
is a tool execution surface, not a hardened public API gateway.

## JupyterLab

The image default starts JupyterLab on port 8888:

```bash
docker run --rm -it -p 8888:8888 -v "$PWD:/work" \
  ghcr.io/argonne-lcf/chemgraph:latest
```

The image starts Jupyter without a token. Publish it only on a trusted machine,
or add authentication for shared deployments.

## Compose profiles

The repository Compose file builds `chemgraph:local`, mounts the checkout at
`/app`, and defines four development profiles:

```bash
docker compose --profile streamlit up --build
docker compose --profile mcp up --build
docker compose --profile jupyter up --build
docker compose --profile cli run --rm cli
```

Compose forwards supported provider variables and stores tool artifacts in the
checkout's `cg_logs/` directory.

## Build and operate safely

```bash
docker build -t chemgraph:local .
```

The TBLite build can take time, especially on ARM. Mount a dedicated artifact
directory rather than a home directory, pass tokens at runtime, remember that
container paths may differ from host paths, and pin an image tag or digest for
reproducible deployments.
