!!! note
      Ensure you have **Python 3.10 or higher** installed on your system.

!!! warning "**Note on Compatibility with conda**"

      - ChemGraph supports both MACE and UMA (Meta's machine learning potential). However, due to the current dependency conflicts, particularly with `e3nn`—**you cannot install both in the same environment**.  
      - To use both libraries, create **separate Conda environments**, one for each.

=== "**install with pip**"

      - Clone the repository:
         ```bash
         git clone https://github.com/Autonomous-Scientific-Agents/ChemGraph
         cd ChemGraph
         ```
      - Create and activate a virtual environment:
         ```bash
         # Using venv (built into Python)
         python -m venv chemgraph-env
         source chemgraph-env/bin/activate  # On Unix/macOS
         # OR
         .\chemgraph-env\Scripts\activate  # On Windows
         ```

      - Install ChemGraph:
         ```bash
         pip install -e .
         ```

=== "**install with conda**"

      - Clone the repository:
         ```bash
         git clone https://github.com/Autonomous-Scientific-Agents/ChemGraph
         cd ChemGraph
         ```
      - Create and activate a new Conda environment:
         ```bash
         conda create -n chemgraph python=3.10 -y
         conda activate chemgraph
         ```
      - Install required Conda dependencies: 
         ```bash
         conda install -c conda-forge nwchem
         ```
      - Install `ChemGraph` and its dependencies:

=== "**install with uv**"

      - Clone the repository:
         ```bash
         git clone https://github.com/Autonomous-Scientific-Agents/ChemGraph
         cd ChemGraph
         ```

      - Create and activate a virtual environment using uv:
         ```bash
         uv venv chemgraph-env
         uv venv --python 3.11 chemgraph-env # For specific python version

         source chemgraph-env/bin/activate # Unix/macos
         .\chemgraph-env\Scripts\activate  # On Windows
         ```

      - Install ChemGraph using uv:
         ```bash
         uv pip install -e .
         ```

!!! note "**Optional: Install with UMA support**"

      - **Note on e3nn Conflict for UMA Installation:** The `uma` extras (requiring `e3nn>=0.5`) conflict with the base `mace-torch` dependency (which pins `e3nn==0.4.4`). 
      - If you need to install UMA support in an environment where `mace-torch` might cause this conflict, you can try the following workaround:
         1. **Temporarily modify `pyproject.toml`**: Open the `pyproject.toml` file in the root of the ChemGraph project.
         2. Find the line containing `"mace-torch>=0.3.13",` in the `dependencies` list.
         3. Comment out this line by adding a `#` at the beginning (e.g., `#    "mace-torch>=0.3.13",`).
         4. **Install UMA extras**: Run `pip install -e ".[uma]"`.
         5. **(Optional) Restore `pyproject.toml`**: After installation, you can uncomment the `mace-torch` line if you still need it for other purposes in the same environment. Be aware that `mace-torch` might not function correctly due to the `e3nn` version mismatch (`e3nn>=0.5` will be present for UMA).
      
      - **The most robust solution for using both MACE and UMA with their correct dependencies is to create separate Conda environments, as highlighted in the "Note on Compatibility" above.**
      
      - **Important for UMA Model Access:** The `facebook/UMA` model is a gated model on Hugging Face. To use it, you must:
         1. Visit the [facebook/UMA model page](https://huggingface.co/facebook/UMA) on Hugging Face.
         2. Log in with your Hugging Face account.
         3. Accept the model's terms and conditions if prompted.
      - Your environment (local or CI) must also be authenticated with Hugging Face, typically by logging in via `huggingface-cli login` or ensuring `HF_TOKEN` is set and recognized.

      ```bash
      pip install -e ".[uma]"
      ```
