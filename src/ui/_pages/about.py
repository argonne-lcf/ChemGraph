"""About ChemGraph page."""

import streamlit as st


def render() -> None:
    """Render the About ChemGraph page."""
    st.title("\U0001f4d6 About ChemGraph")

    st.markdown(
        """
    ## AI Agents for Computational Chemistry
    
    ChemGraph is an **agentic framework** for computational chemistry and materials science workflows. 
    It enables researchers to perform complex computational chemistry tasks using natural language queries 
    powered by large language models (LLMs) and specialized AI agents.
    
    ### \U0001f52c Key Features
    
    - **Multi-Agent Workflows**: Coordinate multiple AI agents for complex computational tasks
    - **Natural Language Interface**: Interact with computational chemistry tools using plain English
    - **Molecular Visualization**: 3D interactive molecular structure visualization
    - **Multiple Calculators**: Support for various quantum chemistry packages (ORCA, Psi4, MACE, etc.)
    - **Report Generation**: Automated generation of computational chemistry reports
    - **Flexible Backends**: Support for various LLM providers (OpenAI, Anthropic, local models)
    
    ### \U0001f4da Resources
    
    - **GitHub**: [https://github.com/argonne-lcf/ChemGraph](https://github.com/argonne-lcf/ChemGraph)
    - **Documentation**: [https://argonne-lcf.github.io/ChemGraph/](https://argonne-lcf.github.io/ChemGraph/)
    
    ### \U0001f3db\ufe0f Developed at Argonne National Laboratory
    
    ChemGraph is developed at **Argonne National Laboratory** as part of advancing 
    computational chemistry and materials science research through AI-driven automation.
    
    ### \U0001f4c4 License
    
    This project is licensed under the **Apache License 2.0** - see the 
    [LICENSE](https://github.com/argonne-lcf/ChemGraph/blob/main/LICENSE) file for details.
    
    ### \U0001f64f Citation
    
    If you use ChemGraph in your research, please cite our [work](https://doi.org/10.1038/s42004-025-01776-9):
    
    ```bibtex
    @article{pham_chemgraph_2026,
    title = {{ChemGraph} as an agentic framework for computational chemistry workflows},
    url = {https://doi.org/10.1038/s42004-025-01776-9},
    doi = {10.1038/s42004-025-01776-9},
    author = {Pham, Thang D. and Tanikanti, Aditya and Ke\\c{c}eli, Murat},
    date = {2026-01-08},
    author={Pham, Thang D and Tanikanti, Aditya and Ke{\\c{c}}eli, Murat},
    journal={Communications Chemistry},
    year={2026},
    publisher={Nature Publishing Group UK London}
    }
    ```

    ### \U0001f64c Acknowledgments

    This research used resources of the Argonne Leadership Computing Facility, a U.S.
    Department of Energy (DOE) Office of Science user facility at Argonne National
    Laboratory and is based on research supported by the U.S. DOE Office of Science-
    Advanced Scientific Computing Research Program, under Contract No. DE-AC02-
    06CH11357. Our work leverages ALCF Inference Endpoints, which provide a robust API
    for LLM inference on ALCF HPC clusters via Globus Compute. We are thankful to Serkan
    Altuntas for his contributions to the user interface of ChemGraph and for insightful
    discussions on AIOps.
    
    ---
    
    ### \U0001f680 Get Started
    
    Ready to use ChemGraph? Switch to the **\U0001f3e0 Main Interface** using the navigation menu on the left 
    to start running computational chemistry workflows with AI agents!
    """
    )
