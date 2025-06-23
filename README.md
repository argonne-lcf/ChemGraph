# ChemGraph

<details>
  <summary><strong>Overview</strong></summary>

**ChemGraph** is an agentic framework that can automate molecular simulation workflows using large language models (LLMs). Built on top of `LangGraph` and `ASE`, ChemGraph allows users to perform complex computational chemistry tasks, from structure generation to thermochemistry calculations, with a natural language interface. 
ChemGraph supports diverse simulation backends, including ab initio quantum chemistry methods (e.g. coupled-cluster, DFT via NWChem, ORCA), semi-empirical methods (e.g., XTB via TBLite), and machine learning potentials (e.g, MACE, UMA) through a modular integration with `ASE`. 

</details>

<details>
  <summary><strong>Installation Instruction</strong></summary>

Ensure you have **Python 3.10 or higher** installed on your system. 
**Using pip (Recommended for most users)**

1. Clone the repository:
   ```bash
   git clone https://github.com/Autonomous-Scientific-Agents/ChemGraph
   cd ChemGraph
    ```
2. Create and activate a virtual environment:
   ```bash
   # Using venv (built into Python)
   python -m venv chemgraph-env
   source chemgraph-env/bin/activate  # On Unix/macOS
   # OR
   .\chemgraph-env\Scripts\activate  # On Windows
   ```

3. Install ChemGraph:
   ```bash
   pip install -e .
   ```

**Using Conda (Alternative)**

> ⚠️ **Note on Compatibility**  
> ChemGraph supports both MACE and UMA (Meta's machine learning potential). However, due to the current dependency conflicts, particularly with `e3nn`—**you cannot install both in the same environment**.  
> To use both libraries, create **separate Conda environments**, one for each.

1. Clone the repository:
   ```bash
   git clone https://github.com/Autonomous-Scientific-Agents/ChemGraph
   cd ChemGraph
    ```
2. Create and activate a new Conda environment:
   ```bash
    conda create -n chemgraph python=3.10 -y
    conda activate chemgraph
    ```
3. Install required Conda dependencies: 
    ```bash
    conda install -c conda-forge nwchem
    ```
4. Install `ChemGraph` and its dependencies:
   
**Optional: Install with UMA support**

> **Note on e3nn Conflict for UMA Installation:** The `uma` extras (requiring `e3nn>=0.5`) conflict with the base `mace-torch` dependency (which pins `e3nn==0.4.4`). 
> If you need to install UMA support in an environment where `mace-torch` might cause this conflict, you can try the following workaround:
> 1. **Temporarily modify `pyproject.toml`**: Open the `pyproject.toml` file in the root of the ChemGraph project.
> 2. Find the line containing `"mace-torch>=0.3.13",` in the `dependencies` list.
> 3. Comment out this line by adding a `#` at the beginning (e.g., `#    "mace-torch>=0.3.13",`).
> 4. **Install UMA extras**: Run `pip install -e ".[uma]"`.
> 5. **(Optional) Restore `pyproject.toml`**: After installation, you can uncomment the `mace-torch` line if you still need it for other purposes in the same environment. Be aware that `mace-torch` might not function correctly due to the `e3nn` version mismatch (`e3nn>=0.5` will be present for UMA).
>
> **The most robust solution for using both MACE and UMA with their correct dependencies is to create separate Conda environments, as highlighted in the "Note on Compatibility" above.**

> **Important for UMA Model Access:** The `facebook/UMA` model is a gated model on Hugging Face. To use it, you must:
> 1. Visit the [facebook/UMA model page](https://huggingface.co/facebook/UMA) on Hugging Face.
> 2. Log in with your Hugging Face account.
> 3. Accept the model's terms and conditions if prompted.
> Your environment (local or CI) must also be authenticated with Hugging Face, typically by logging in via `huggingface-cli login` or ensuring `HF_TOKEN` is set and recognized.

```bash
pip install -e ".[uma]"
```
</details>

<details>
  <summary><strong>Example Usage</strong></summary>

1. Before exploring example usage in the `notebooks/` directory, ensure you have specified the necessary API tokens in your environment. For example, you can set the OpenAI API token and Anthropic API token using the following commands:

   ```bash
   # Set OpenAI API token
   export OPENAI_API_KEY="your_openai_api_key_here"

   # Set Anthropic API token
   export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
   ```

2. **Explore Example Notebooks**: Navigate to the `notebooks/` directory to explore various example notebooks demonstrating different capabilities of ChemGraph.

   - **[Single-Agent System with MACE](notebooks/Demo_single_agent.ipynb)**: This notebook demonstrates how a single agent can utilize multiple tools with MACE/xTB support.

   - **[Single-Agent System with UMA](notebooks/Demo_single_agent_UMA.ipynb)**: This notebook demonstrates how a single agent can utilize multiple tools with UMA support.

   - **[Multi-Agent System](notebooks/Demo_multi_agent.ipynb)**: This notebook demonstrates a multi-agent setup where different agents (Planner, Executor and Aggregator) handle various tasks exemplifying the collaborative potential of ChemGraph.

   - **[Single-Agent System with gRASPA](notebooks/Demo_graspa_agent.ipynb)**: This notebook provides a sample guide on executing a gRASPA simulation using a single agent. For gRASPA-related installation instructions, visit the [gRASPA GitHub repository](https://github.com/snurr-group/gRASPA). The notebook's functionality has been validated on a single compute node at ALCF Polaris.

</details>

<details>
  <summary><strong>Project Structure</strong></summary>

```
chemgraph/
│
├── src/                       # Source code
│   ├── chemgraph/             # Top-level package
│   │   ├── agent/             # Agent-based task management
│   │   ├── graphs/            # Workflow graph utilities
│   │   ├── models/            # Different Pydantic models
│   │   ├── prompt/            # Agent prompt
│   │   ├── state/             # Agent state
│   │   ├── tools/             # Tools for molecular simulations
│   │   ├── utils/             # Other utility functions
│
├── pyproject.toml             # Project configuration
└── README.md                  # Project documentation
```

</details>

<details>
  <summary><strong>Running Local Models with vLLM</strong></summary>
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
</details>

<details>
  <summary><strong>Docker Support with Docker Compose (Recommended for vLLM)</strong></summary>

This project uses Docker Compose to manage multi-container applications, providing a consistent development and deployment environment. This setup allows you to run the `chemgraph` (with JupyterLab) and a local vLLM model server as separate, inter-communicating services.

**Prerequisites**

- [Docker](https://docs.docker.com/get-docker/) installed on your system.
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.
- [vllm](https://github.com/vllm-project/vllm) cloned into the project root. `git clone https://github.com/vllm-project/vllm.git`

**Overview**

The `docker-compose.yml` file defines two main services:
1.  **`jupyter_lab`**: 
    *   Builds from the main `Dockerfile`.
    *   Runs JupyterLab, allowing you to interact with the notebooks and agent code.
    *   Is configured to communicate with the `vllm_server`.
2.  **`vllm_server`**:
    *   Builds from `Dockerfile.arm` by default (located in the project root), which is suitable for running vLLM on macOS (Apple Silicon / ARM-based CPUs). This Dockerfile is a modified version intended for CPU execution.
    *   For other operating systems or hardware (e.g., Linux with NVIDIA GPUs), you will need to use a different Dockerfile. The vLLM project provides a collection of Dockerfiles for various architectures (CPU, CUDA, ROCm, etc.) available at [https://github.com/vllm-project/vllm/tree/main/docker](https://github.com/vllm-project/vllm/tree/main/docker). You would need to adjust the `docker-compose.yml` to point to the appropriate Dockerfile and context (e.g., by cloning the vLLM repository locally and referencing a Dockerfile within it).
    *   Starts an OpenAI-compatible API server using vLLM, serving a pre-configured model (e.g., `meta-llama/Llama-3-8B-Instruct` as per the current `docker-compose.yml`).
    *   Listens on port 8000 within the Docker network (and is exposed to host port 8001 by default).

**Building and Running with Docker Compose**

Navigate to the root directory of the project (where `docker-compose.yml` is located) and run:

```bash
docker-compose up --build
```

**Note on Hugging Face Token (`HF_TOKEN`):**
Many models, including the default `meta-llama/Llama-3-8B-Instruct`, are gated and require Hugging Face authentication. To provide your Hugging Face token to the `vllm_server` service:

1.  **Create a `.env` file** in the root directory of the project (the same directory as `docker-compose.yml`).
2.  Add your Hugging Face token to this file:
    ```
    HF_TOKEN="your_actual_hugging_face_token_here"
    ```
    
Docker Compose will automatically load this variable when you run `docker-compose up`. The `vllm_server` in `docker-compose.yml` is configured to use this environment variable.

Breakdown of the command:
- `docker-compose up`: Starts or restarts all services defined in `docker-compose.yml`.
- `--build`: Forces Docker Compose to build the images before starting the containers. This is useful if you've made changes to `Dockerfile`, `Dockerfile.arm` (or other vLLM Dockerfiles), or project dependencies.

After running this command:
- The vLLM server will start, and its logs will be streamed to your terminal.
- JupyterLab will start, and its logs will also be streamed. JupyterLab will be accessible in your web browser at `http://localhost:8888`. No token is required by default.

To stop the services, press `Ctrl+C` in the terminal where `docker-compose up` is running. To stop and remove the containers, you can use `docker-compose down`.

### Configuring Notebooks to Use the Local vLLM Server

When you initialize `ChemGraph` in your Jupyter notebooks (running within the `jupyter_lab` service), you can now point to the local vLLM server:

1.  **Model Name**: Use the Hugging Face identifier of the model being served by vLLM (e.g., `meta-llama/Llama-3-8B-Instruct` as per default in `docker-compose.yml`).
2.  **Base URL & API Key**: These are automatically passed as environment variables (`VLLM_BASE_URL` and `OPENAI_API_KEY`) to the `jupyter_lab` service by `docker-compose.yml`. The agent code in `llm_agent.py` has been updated to automatically use these environment variables if a model name is provided that isn't in the pre-defined supported lists (OpenAI, Ollama, ALCF, Anthropic).

**Example in a notebook:**

```python
from chemgraph.agent.llm_agent import ChemGraph

# The model name should match what vLLM is serving.
# The base_url and api_key will be picked up from environment variables
# set in docker-compose.yml if this model_name is not a standard one.
agent = ChemGraph(
    model_name="meta-llama/Llama-3-8B-Instruct", # Or whatever model is configured in docker-compose.yml
    workflow_type="single_agent", 
    # No need to explicitly pass base_url or api_key here if using the docker-compose setup
)

# Now you can run the agent
# response = agent.run("What is the SMILES string for water?")
# print(response)
```

The `jupyter_lab` service will connect to `http://vllm_server:8000/v1` (as defined by `VLLM_BASE_URL` in `docker-compose.yml`) to make requests to the language model.

### GPU Support for vLLM (Advanced)

The provided `Dockerfile.arm` and the default `docker-compose.yml` setup are configured for CPU-based vLLM (suitable for macOS). To enable GPU support (typically on Linux with NVIDIA GPUs):

1.  **Choose the Correct vLLM Dockerfile**:
    *   Do **not** use `Dockerfile.arm`.
    *   You will need to use a Dockerfile from the official vLLM repository designed for CUDA. Clone the vLLM repository (e.g., into a `./vllm` subdirectory in your project) or use it as a submodule.
    *   A common choice is `vllm/docker/Dockerfile` (for CUDA) or a specific version like `vllm/docker/Dockerfile.cuda-12.1`. Refer to [vLLM Dockerfiles](https://github.com/vllm-project/vllm/tree/main/docker) for options.
2.  **Modify `docker-compose.yml`**:
    *   Change the `build.context` for the `vllm_server` service to point to your local clone of the vLLM repository (e.g., `./vllm`).
    *   Change the `build.dockerfile` to the path of the CUDA-enabled Dockerfile within that context (e.g., `docker/Dockerfile`).
    *   Uncomment and configure the `deploy.resources.reservations.devices` section for the `vllm_server` service to grant it GPU access.

    ```yaml
    # ... in docker-compose.yml, for vllm_server:
    # build:
    #   context: ./vllm  # Path to your local vLLM repo clone
    #   dockerfile: docker/Dockerfile # Path to the CUDA Dockerfile within the vLLM repo
    # ...
    # environment:
      # Remove or comment out:
      # - VLLM_CPU_ONLY=1 
      # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1 # or 'all'
              capabilities: [gpu]
    ```
3.  **NVIDIA Container Toolkit**: Ensure you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on your host system for Docker to recognize and use NVIDIA GPUs.
4.  **Build Arguments**: Some official vLLM Dockerfiles accept build arguments (e.g., `CUDA_VERSION`, `PYTHON_VERSION`). You might need to pass these via the `build.args` section in `docker-compose.yml`.

    ```yaml
    # ... in docker-compose.yml, for vllm_server build:
    # args:
    #   - CUDA_VERSION=12.1.0 
    #   - PYTHON_VERSION=3.10 
    ```
    Consult the specific vLLM Dockerfile you choose for available build arguments.

### Running Only JupyterLab (for External LLM Services)

If you prefer to use external LLM services like OpenAI, Claude, or other hosted providers instead of running a local vLLM server, you can run only the JupyterLab service:

```bash
docker-compose up jupyter_lab
```

This will start only the JupyterLab container without the vLLM server. In this setup:

1. **JupyterLab Access**: JupyterLab will be available at `http://localhost:8888`
2. **LLM Configuration**: In your notebooks, configure the agent to use external services by providing appropriate model names and API keys:

**Example for OpenAI:**
```python
import os
from chemgraph.agent.llm_agent import ChemGraph

# Set your OpenAI API key as an environment variable or pass it directly
os.environ["OPENAI_API_KEY"] = "your-openai-api-key-here"

agent = ChemGraph(
    model_name="gpt-4",  # or "gpt-3.5-turbo", "gpt-4o", etc.
    workflow_type="single_agent"
)
```

**Example for Anthropic Claude:**
```python
import os
from chemgraph.agent.llm_agent import ChemGraph

# Set your Anthropic API key
os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-api-key-here"

agent = ChemGraph(
    model_name="claude-3-sonnet-20240229",  # or other Claude models
    workflow_type="single_agent_ase"
)
```

**Available Environment Variables for External Services:**
- `OPENAI_API_KEY`: For OpenAI models
- `ANTHROPIC_API_KEY`: For Anthropic Claude models

### Working with Example Notebooks

Once JupyterLab is running (via `docker-compose up` or `docker-compose up jupyter_lab`), you can navigate to the `notebooks/` directory within the JupyterLab interface to open and run the example notebooks. Modify them as shown above to use either the locally served vLLM model or external LLM services.

### Notes on TBLite Python API

The `tblite` package is installed via pip within the `jupyter_lab` service. For the full Python API functionality of TBLite (especially for XTB), you might need to follow separate installation instructions as mentioned in the [TBLite documentation](https://tblite.readthedocs.io/en/latest/installation.html). If you require this, you may need to modify the main `Dockerfile` to include these additional installation steps or perform them inside a running container and commit the changes to a new image for the `jupyter_lab` service.

</details>

<details>
  <summary><strong>Code Formatting & Linting</strong></summary>

This project uses [Ruff](https://github.com/astral-sh/ruff) for **both formatting and linting**. To ensure all code follows our style guidelines, install the pre-commit hook:

```sh
pip install pre-commit
pre-commit install
```
</details>

<details>
  <summary><strong>License</strong></summary>
This project is licensed under the Apache 2.0 License.
</details>