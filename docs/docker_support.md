!!! note
    This project uses Docker Compose to manage multi-container applications, providing a consistent development and deployment environment. This setup allows you to run the `chemgraph` (with JupyterLab) and a local vLLM model server as separate, inter-communicating services.

### **Prerequisites**

- [Docker](https://docs.docker.com/get-docker/) installed on your system.
- [Docker Compose](https://docs.docker.com/compose/install/) installed on your system.
- [vllm](https://github.com/vllm-project/vllm) cloned into the project root. `git clone https://github.com/vllm-project/vllm.git`

### **Overview**

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
- `GEMINI_API_KEY`: For Gemini models

### Working with Example Notebooks

Once JupyterLab is running (via `docker-compose up` or `docker-compose up jupyter_lab`), you can navigate to the `notebooks/` directory within the JupyterLab interface to open and run the example notebooks. Modify them as shown above to use either the locally served vLLM model or external LLM services.

### Notes on TBLite Python API

The `tblite` package is installed via pip within the `jupyter_lab` service. For the full Python API functionality of TBLite (especially for XTB), you might need to follow separate installation instructions as mentioned in the [TBLite documentation](https://tblite.readthedocs.io/en/latest/installation.html). If you require this, you may need to modify the main `Dockerfile` to include these additional installation steps or perform them inside a running container and commit the changes to a new image for the `jupyter_lab` service.