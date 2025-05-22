# comp_chem_agent

## Description

`comp_chem_agent` is a computational chemistry agent designed to assist with **molecular simulation tasks**. The package integrates key libraries such as `ASE`, `RDKit`, and modern agent frameworks like `LangGraph` and `LangChain`. This tool simplifies molecular modeling workflows, enabling seamless interaction with high-performance computing (HPC) environments.

---

## Features

- **Molecular Simulations**: Leverages `ASE` and `RDKit` for molecular structure handling and simulation workflows.
- **Agent-Based Tasks**: Uses `LangGraph` and `LangChain` to coordinate tasks dynamically.
- **HPC Integration**: Designed for computational chemistry tasks on HPC systems.

---

## Installation

Ensure Python 3.10 or above is installed on your system.

1. Clone the repository:

   ```bash
   git clone https://github.com/Autonomous-Scientific-Agents/CompChemAgent.git
   cd CompChemAgent
   ```

2. Install the package using `pip`:

   ```bash
   pip install .
   ```

---

## Dependencies

The following libraries are required and will be installed automatically:

- `ase==3.22.1`
- `rdkit==2024.03.5`
- `langgraph==0.2.59`
- `langchain-openai==0.2.12`
- `langchain-ollama==0.2.1`
- `pydantic==2.10.3`
- `pandas>=2.2`
- `mace-torch==0.3.9`
- `torch<2.6`
- `torch-dftd==0.5.1`
- `pubchempy==1.0.4`
- `pyppeteer==2.0.0`
- `numpy<2`
- `qcelemental==0.29.0`
- `qcengine==0.31.0`
- `tblite==0.4.0`

For TBLite (for XTB), to use the Python extension, you must install it separately. Instructions to install Python API for TBLite can be found here: https://tblite.readthedocs.io/en/latest/installation.html
---

## Usage

Explore example workflows in the notebooks/ directory:

Single-Agent System: Demo-SingleAgent.ipynb
- Demonstrates a basic agent with multiple tools.

Multi-Agent System: Demo_MultiAgent.ipynb
- Demonstrates multiple agents handling different tasks.

Legacy Implementation: Legacy-ComChemAgent.ipynb
- Uses deprecated create_react_agent method in LangGraph.

---

## Running Local Models with vLLM

This section describes how to set up and run local language models using the vLLM inference server.

### Inference Backend Setup (Remote/Local)

#### Virtual Python Environment
All instructions below must be executed within a Python virtual environment. Ensure the virtual environment uses the same Python version as your project (e.g., Python 3.11).

**Example 1: Using conda**
```bash
conda create -n vllm-env python=3.11 -y
conda activate vllm-env
```

**Example 2: Using python venv**
```bash
python3.11 -m venv vllm-env
source vllm-env/bin/activate  # On Windows use `vllm-env\\Scripts\\activate`
```

#### Install Inference Server (vLLM)
vLLM is recommended for serving many transformer models efficiently.

**Basic vLLM installation from source:**
Make sure your virtual environment is activated.
```bash
# Ensure git is installed
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```
For specific hardware acceleration (e.g., CUDA, ROCm), refer to the [official vLLM installation documentation](https://docs.vllm.ai/en/latest/getting_started/installation.html).

#### Running the vLLM Server (Standalone)

A script is provided at `scripts/run_vllm_server.sh` to help start a vLLM server with features like logging, retry attempts, and timeout. This is useful for running vLLM outside of Docker Compose, for example, directly on a machine with GPU access.

**Before running the script:**
1.  Ensure your vLLM Python virtual environment is activated.
    ```bash
    # Example: if you used conda
    # conda activate vllm-env 
    # Example: if you used python venv
    # source path/to/your/vllm-env/bin/activate
    ```
2.  Make the script executable:
    ```bash
    chmod +x scripts/run_vllm_server.sh
    ```

**To run the script:**

```bash
./scripts/run_vllm_server.sh [MODEL_IDENTIFIER] [PORT] [MAX_MODEL_LENGTH]
```

-   `[MODEL_IDENTIFIER]` (optional): The Hugging Face model identifier. Defaults to `facebook/opt-125m`.
-   `[PORT]` (optional): The port for the vLLM server. Defaults to `8001`.
-   `[MAX_MODEL_LENGTH]` (optional): The maximum model length. Defaults to `4096`.

**Example:**
```bash
./scripts/run_vllm_server.sh meta-llama/Meta-Llama-3-8B-Instruct 8001 8192
```

**Important Note on Gated Models (e.g., Llama 3):**
Many models, such as those from the Llama family by Meta, are gated and require you to accept their terms of use on Hugging Face and use an access token for download. 

To use such models with vLLM (either via the script or Docker Compose):
1.  **Hugging Face Account and Token**: Ensure you have a Hugging Face account and have generated an access token with `read` permissions. You can find this in your Hugging Face account settings under "Access Tokens".
2.  **Accept Model License**: Navigate to the Hugging Face page of the specific model you want to use (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) and accept its license/terms if prompted.
3.  **Environment Variables**: Before running the vLLM server (either via the script or `docker-compose up`), you need to set the following environment variables in your terminal session or within your environment configuration (e.g., `.bashrc`, `.zshrc`, or by passing them to Docker Compose if applicable):
    ```bash
    export HF_TOKEN="your_hugging_face_token_here"
    # Optional: Specify a directory for Hugging Face to download models and cache.
    # export HF_HOME="/path/to/your/huggingface_cache_directory"
    ```
    vLLM will use these environment variables to authenticate with Hugging Face and download the model weights.

The script will:
- Attempt to start the vLLM OpenAI-compatible API server.
- Log output to a file in the `logs/` directory (created if it doesn't exist at the project root).
- The server runs in the background via `nohup`.

This standalone script is an alternative to running vLLM via Docker Compose and is primarily for users who manage their vLLM instances directly.

---

## Project Structure

```
comp_chem_agent/
│
├── src/                       # Source code
│   ├── comp_chem_agent/       # Top-level package
│   │   ├── agent/             # Agent-based task management
│   │   ├── graphs/            # Workflow graph utilities
│   │   ├── models/            # Different Pydantic models
│   │   ├── prompt/            # Agent prompt
│   │   ├── state/             # Agent state
│   │   ├── tools/             # Tools for molecular simulations
│
├── pyproject.toml             # Project configuration
└── README.md                  # Project documentation
```

---

## Code Formatting & Linting

This project uses [Ruff](https://github.com/astral-sh/ruff) for **both formatting and linting**.

### **Setup**
To ensure all code follows our style guidelines, install the pre-commit hook:

```sh
pip install pre-commit
pre-commit install
```

---

## Docker Support with Docker Compose (Recommended for vLLM)

This project uses Docker Compose to manage multi-container applications, providing a consistent development and deployment environment. This setup allows you to run the `comp_chem_agent` (with JupyterLab) and a local vLLM model server as separate, inter-communicating services.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your system.
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.

### Overview

The `docker-compose.yml` file defines two main services:
1.  **`jupyter_lab`**: 
    *   Builds from the main `Dockerfile`.
    *   Runs JupyterLab, allowing you to interact with the notebooks and agent code.
    *   Is configured to communicate with the `vllm_server`.
2.  **`vllm_server`**:
    *   Builds from `vllm.Dockerfile`.
    *   Clones and installs vLLM (CPU version by default).
    *   Starts an OpenAI-compatible API server using vLLM, serving a pre-configured model (e.g., `facebook/opt-125m`).
    *   Listens on port 8000 within the Docker network (and can be exposed to host port 8001).

### Building and Running with Docker Compose

Navigate to the root directory of the project (where `docker-compose.yml` is located) and run:

```bash
docker-compose up --build
```

Breakdown of the command:
- `docker-compose up`: Starts or restarts all services defined in `docker-compose.yml`.
- `--build`: Forces Docker Compose to build the images before starting the containers. This is useful if you've made changes to `Dockerfile`, `vllm.Dockerfile`, or project dependencies.

After running this command:
- The vLLM server will start, and its logs will be streamed to your terminal.
- JupyterLab will start, and its logs will also be streamed. JupyterLab will be accessible in your web browser at `http://localhost:8888`. No token is required by default.

To stop the services, press `Ctrl+C` in the terminal where `docker-compose up` is running. To stop and remove the containers, you can use `docker-compose down`.

### Configuring Notebooks to Use the Local vLLM Server

When you initialize `CompChemAgent` or `llm_graph` in your Jupyter notebooks (running within the `jupyter_lab` service), you can now point to the local vLLM server:

1.  **Model Name**: Use the Hugging Face identifier of the model being served by vLLM (e.g., `facebook/opt-125m` as per default in `vllm.Dockerfile` and `docker-compose.yml`).
2.  **Base URL & API Key**: These are automatically passed as environment variables (`VLLM_BASE_URL` and `OPENAI_API_KEY`) to the `jupyter_lab` service by `docker-compose.yml`. The agent code in `llm_graph.py` and `llm_agent.py` has been updated to automatically use these environment variables if a model name is provided that isn't in the pre-defined supported lists (OpenAI, Ollama, ALCF, Anthropic).

**Example in a notebook:**

```python
from comp_chem_agent.agent import llm_graph # Or CompChemAgent

# The model name should match what vLLM is serving.
# The base_url and api_key will be picked up from environment variables
# set in docker-compose.yml if this model_name is not a standard one.
agent = llm_graph.llm_graph(
    model_name="facebook/opt-125m", # Or whatever model is configured in vllm.Dockerfile/docker-compose
    workflow_type="single_agent_ase", 
    # No need to explicitly pass base_url or api_key here if using the docker-compose setup
)

# Now you can run the agent
# response = agent.run("What is the SMILES string for water?")
# print(response)
```

The `jupyter_lab` service will connect to `http://vllm_server:8000/v1` (as defined by `VLLM_BASE_URL` in `docker-compose.yml`) to make requests to the language model.

### GPU Support for vLLM (Advanced)

The provided `vllm.Dockerfile` and `docker-compose.yml` are configured for CPU-based vLLM. To enable GPU support:

1.  **Base Image for `vllm.Dockerfile`**: You'll need to change the base image in `vllm.Dockerfile` to one that includes CUDA drivers (e.g., an official NVIDIA CUDA image like `nvidia/cuda:12.1.0-devel-ubuntu22.04`).
2.  **vLLM Installation**: Ensure vLLM is installed with GPU support. This usually happens automatically if CUDA is detected during `pip install -e .`.
3.  **`docker-compose.yml`**: Uncomment and configure the `deploy.resources.reservations.devices` section for the `vllm_server` service to grant it GPU access.

    ```yaml
    # ... inside vllm_server service definition ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 # or 'all'
              capabilities: [gpu]
    ```
    You may need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your host system for Docker to recognize GPUs.

### Working with Example Notebooks

Once JupyterLab is running (via `docker-compose up`), you can navigate to the `notebooks/` directory within the JupyterLab interface to open and run the example notebooks. Modify them as shown above to use the locally served vLLM model.

### Notes on TBLite Python API

The `tblite` package is installed via pip within the `jupyter_lab` service. For the full Python API functionality of TBLite (especially for XTB), you might need to follow separate installation instructions as mentioned in the [TBLite documentation](https://tblite.readthedocs.io/en/latest/installation.html). If you require this, you may need to modify the main `Dockerfile` to include these additional installation steps or perform them inside a running container and commit the changes to a new image for the `jupyter_lab` service.

---

## License

This project is licensed under the MIT License.
