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



## License

This project is licensed under the MIT License.
