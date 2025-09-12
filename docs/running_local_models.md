!!! note
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

???+ info "**Important Note on Gated Models (e.g., Llama 3):**"
    - Many models, such as those from the Llama family by Meta, are gated and require you to accept their terms of use on Hugging Face and use an access token for download. 

    - To use such models with vLLM (either via the script or Docker Compose):
        1. **Hugging Face Account and Token**: Ensure you have a Hugging Face account and have generated an access token with `read` permissions. You can find this in your Hugging Face account settings under "Access Tokens".
        2.  **Accept Model License**: Navigate to the Hugging Face page of the specific model you want to use (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) and accept its license/terms if prompted.
        3.  **Environment Variables**: Before running the vLLM server (either via the script or `docker-compose up`), you need to set the following environment variables in your terminal session or within your environment configuration (e.g., `.bashrc`, `.zshrc`, or by passing them to Docker Compose if applicable):
            ```bash
            export HF_TOKEN="your_hugging_face_token_here"
            # Optional: Specify a directory for Hugging Face to download models and cache.
            # export HF_HOME="/path/to/your/huggingface_cache_directory"
            ```
            vLLM will use these environment variables to authenticate with Hugging Face and download the model weights.

    - The script will:
        - Attempt to start the vLLM OpenAI-compatible API server.
        - Log output to a file in the `logs/` directory (created if it doesn't exist at the project root).
        - The server runs in the background via `nohup`.

    - This standalone script is an alternative to running vLLM via Docker Compose and is primarily for users who manage their vLLM instances directly.