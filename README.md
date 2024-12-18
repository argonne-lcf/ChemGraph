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

- `pydantic>=1.8.2`
- `ase>=3.22.0`
- `rdkit>=2024.03.5`
- `langgraph>=0.2.59`
- `langchain-openai>=0.2.12`
- `langchain-ollama>=0.2.1`
- `pydantic>=2.10.3`
- `pandas>=2.2`

---

## Usage

After installation, you can import and use the package in your scripts:

```python
from comp_chem_agent.agent.llm_agent import *
cca = CompChemAgent()
cca.run("Run geometry optimization using ASE for the molecule with the smiles c1ccccc1 using your available tools.")
```

Another current usage is to create simulation input file for packages such as RASPA2.

```python
from comp_chem_agent.agent.llm_agent import *
cca = CompChemAgent()
mess = cca.return_input("Create a simulation input file to calculate H2 adsorption in a MOF named IRMOF1.cif at 77K and 100 bar using a 2 3 4 unit cell")
print(mess)
```

New update includes the LangGraph workflow for the geometry optimization. Example usage:

```python
from comp_chem_agent.agent.llm_graph import *
graph = llm_graph().geo_opt_graph()

user_input = "Run geometry optimization for the molecule with the smiles c1ccccc1 using your available tools."
config = {"configurable": {"thread_id": "1"}}

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]}, config, stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()
```
---

## Project Structure

```
comp_chem_agent/
│
├── src/                       # Source code
│   ├── comp_chem_agent/       # Top-level package
│   │   ├── tools/             # Tools for molecular simulations
│   │   ├── agent/             # Agent-based task management
│   │   ├── models/            # Different Pydantic models
│   │   └── graphs/            # Workflow graph utilities
│
├── pyproject.toml             # Project configuration
└── README.md                  # Project documentation
```

---

## License

This project is licensed under the MIT License.
