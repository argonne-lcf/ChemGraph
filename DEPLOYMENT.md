# ChemGraph Deployment Guide

This guide describes how to deploy ChemGraph using Docker. The deployment supports both an interactive Command Line Interface (CLI) and a JupyterLab environment.

## Prerequisites

- Docker
- Docker Compose (v2 or later recommended) (Optional, but easier)
- API Keys for Large Language Models (LLMs) (e.g., OpenAI, Anthropic, Gemini, Groq)

## Building the Docker Image

To build the Docker image locally:

```bash
docker build -t chemgraph:latest .
```

## Running the Application

### Option 1: Using Docker Compose (Recommended)

This method simplifies managing environment variables and volumes.

1.  **Configure API Keys**: Create a `.env` file in the root directory (or ensure your environment variables are set in your shell).
    ```env
    OPENAI_API_KEY=your_key_here
    ANTHROPIC_API_KEY=your_key_here
    GEMINI_API_KEY=your_key_here
    GROQ_API_KEY=your_key_here
    ```

2.  **Run CLI Interactively**:
    ```bash
    docker-compose run --rm chemgraph-cli
    ```
    This drops you into the ChemGraph interactive shell.

3.  **Run JupyterLab**:
    ```bash
    docker-compose up chemgraph-jupyter
    ```
    Access JupyterLab at `http://localhost:8888`.

### Option 2: Using Docker CLI Directly

1.  **Run CLI Interactively**:
    ```bash
    docker run -it --rm \
      -v "$(pwd):/app" \
      -e OPENAI_API_KEY=$OPENAI_API_KEY \
      chemgraph:latest
    ```
    *Note: Add other API keys as `-e VAR_NAME=value` flags as needed.*

2.  **Run JupyterLab**:
    ```bash
    docker run -it --rm \
      -p 8888:8888 \
      -v "$(pwd):/app" \
      chemgraph:latest \
      jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --LabApp.token=''
    ```

## Development

The `Dockerfile` and `docker-compose.yml` map the local directory to `/app` in the container. This means changes you make to the code locally are immediately visible in the container (for Python code, thanks to `-e .` editable install behavior, though `Dockerfile` uses standard install, mapping the volume overlays the source code).

### Rebuilding Dependencies

If you change `pyproject.toml` or `environment.yml`, you need to rebuild the image:

```bash
docker-compose build
```
