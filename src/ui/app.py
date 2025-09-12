import streamlit as st

# Page configuration -- MUST be first Streamlit call
st.set_page_config(
    page_title="ChemGraph",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

import json
import ast
import toml
import os
from io import StringIO
from uuid import uuid4
import re
from typing import Optional, Dict, Any
from pathlib import Path
import base64

# Third-party imports
import numpy as np
import pandas as pd
from ase import Atoms
from ase.data import chemical_symbols

# ChemGraph imports
from chemgraph.tools.ase_tools import (
    create_ase_atoms,
    create_xyz_string,
    extract_ase_atoms_from_tool_result,
)
from chemgraph.models.supported_models import all_supported_models

# Configuration management
try:
    from .config import load_config, save_config, get_default_config, flatten_config
except ImportError:
    # Handle case when running as script (not as package)
    import sys
    import os

    # Get current directory - handle both package and script execution
    if "__file__" in globals():
        current_dir = os.path.dirname(os.path.abspath(__file__))
    else:
        current_dir = os.getcwd()

    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)

    try:
        from config import load_config, save_config, get_default_config, flatten_config
    except ImportError:
        # Fallback: assume we're in the project root
        config_dir = os.path.join(os.getcwd(), "src", "ui")
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
        from config import load_config, save_config, get_default_config, flatten_config


# -----------------------------------------------------------------------------
# Optional 3-D viewer - stmol + py3Dmol
# -----------------------------------------------------------------------------
try:
    import stmol

    # Check if stmol works by testing a simple import
    from stmol import showmol

    STMOL_AVAILABLE = True
    st.info("3D visualization is available via stmol.")
except ImportError as e:
    STMOL_AVAILABLE = False
    st.warning("⚠️ **stmol** not available – falling back to text/table view.")
    st.info("To enable 3D visualization, install with: `pip install stmol`")

# -----------------------------------------------------------------------------
# Page Navigation
# -----------------------------------------------------------------------------
st.sidebar.title("🧪 ChemGraph")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Main Interface", "⚙️ Configuration", "📖 About ChemGraph"],
    index=0,
    key="page_navigation",
)

# -----------------------------------------------------------------------------
# About Page
# -----------------------------------------------------------------------------
if page == "📖 About ChemGraph":
    st.title("📖 About ChemGraph")

    st.markdown(
        """
    ## AI Agents for Computational Chemistry
    
    ChemGraph is an **agentic framework** for computational chemistry and materials science workflows. 
    It enables researchers to perform complex computational chemistry tasks using natural language queries 
    powered by large language models (LLMs) and specialized AI agents.
    
    ### 🔬 Key Features
    
    - **Multi-Agent Workflows**: Coordinate multiple AI agents for complex computational tasks
    - **Natural Language Interface**: Interact with computational chemistry tools using plain English
    - **Molecular Visualization**: 3D interactive molecular structure visualization
    - **Multiple Calculators**: Support for various quantum chemistry packages (ORCA, Psi4, MACE, etc.)
    - **Report Generation**: Automated generation of computational chemistry reports
    - **Flexible Backends**: Support for various LLM providers (OpenAI, Anthropic, local models)
    
    ### 📚 Resources
    
    #### 🐙 GitHub Repository
    **Source Code & Documentation**  
    [https://github.com/argonne-lcf/ChemGraph](https://github.com/argonne-lcf/ChemGraph)
    
    - ⭐ Star the repository to stay updated
    - 📝 Submit issues and feature requests
    - 🤝 Contribute to the open-source project
    - 📖 Access detailed documentation and examples
    
    #### 📄 Research Paper
    **ArXiv Preprint**  
    [https://arxiv.org/abs/2506.06363](https://arxiv.org/abs/2506.06363)
    
    - 🔬 Read about the scientific methodology
    - 📊 View benchmark results and case studies
    - 🎯 Understand the technical architecture
    - 📋 Cite this work in your research
    
    ### 🏛️ Developed at Argonne National Laboratory
    
    ChemGraph is developed at **Argonne National Laboratory** as part of advancing 
    computational chemistry and materials science research through AI-driven automation.
    
    ### 📄 License
    
    This project is licensed under the **Apache License 2.0** - see the 
    [LICENSE](https://github.com/argonne-lcf/ChemGraph/blob/main/LICENSE) file for details.
    
    ### 🙏 Citation
    
    If you use ChemGraph in your research, please cite our work:
    
    ```bibtex
    @article{pham2025chemgraph,
    title={ChemGraph: An Agentic Framework for Computational Chemistry Workflows},
    author={Pham, Thang D and Tanikanti, Aditya and Keçeli, Murat},
    journal={arXiv preprint arXiv:2506.06363},
    year={2025}
    url={https://arxiv.org/abs/2506.06363}
    }
    ```
    
    ---
    
    ### 🚀 Get Started
    
    Ready to use ChemGraph? Switch to the **🏠 Main Interface** using the navigation menu on the left 
    to start running computational chemistry workflows with AI agents!
    """
    )

    # Stop execution here for About page
    st.stop()

# -----------------------------------------------------------------------------
# Configuration Page
# -----------------------------------------------------------------------------
elif page == "⚙️ Configuration":
    st.title("⚙️ Configuration")
    st.markdown(
        """
    Edit and manage your ChemGraph configuration settings. Changes are saved to `config.toml`.
    """
    )

    # Initialize session state for config
    if "config" not in st.session_state:
        st.session_state.config = load_config()

    config = st.session_state.config

    # Configuration tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔧 General Settings", "🔗 API Settings", "📝 Raw TOML"]
    )

    with tab1:
        st.subheader("General Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Model & Workflow**")
            config["general"]["model"] = st.selectbox(
                "Model",
                all_supported_models,
                index=(
                    all_supported_models.index(config["general"]["model"])
                    if config["general"]["model"] in all_supported_models
                    else 0
                ),
                key="config_model",
            )

            config["general"]["workflow"] = st.selectbox(
                "Workflow",
                ["single_agent", "multi_agent", "python_repl", "graspa"],
                index=(
                    ["single_agent", "multi_agent", "python_repl", "graspa"].index(
                        config["general"]["workflow"]
                    )
                    if config["general"]["workflow"]
                    in ["single_agent", "multi_agent", "python_repl", "graspa"]
                    else 0
                ),
                key="config_workflow",
            )

            config["general"]["output"] = st.selectbox(
                "Output Format",
                ["state", "last_message"],
                index=(
                    ["state", "last_message"].index(config["general"]["output"])
                    if config["general"]["output"] in ["state", "last_message"]
                    else 0
                ),
                key="config_output",
            )

            config["general"]["structured"] = st.checkbox(
                "Structured Output",
                value=config["general"]["structured"],
                key="config_structured",
            )

            config["general"]["report"] = st.checkbox(
                "Generate Report",
                value=config["general"]["report"],
                key="config_report",
            )

            config["general"]["verbose"] = st.checkbox(
                "Verbose Output",
                value=config["general"]["verbose"],
                key="config_verbose",
            )

        with col2:
            st.write("**Execution Settings**")
            config["general"]["thread"] = st.number_input(
                "Thread ID",
                min_value=1,
                max_value=1000,
                value=config["general"]["thread"],
                key="config_thread",
            )

            config["general"]["recursion_limit"] = st.number_input(
                "Recursion Limit",
                min_value=1,
                max_value=100,
                value=config["general"]["recursion_limit"],
                key="config_recursion",
            )

        st.subheader("Chemistry Settings")

        col3, col4 = st.columns(2)

        with col3:
            st.write("**Optimization**")
            config["chemistry"]["optimization"]["method"] = st.selectbox(
                "Method",
                ["BFGS", "L-BFGS-B", "CG", "Newton-CG"],
                index=(
                    ["BFGS", "L-BFGS-B", "CG", "Newton-CG"].index(
                        config["chemistry"]["optimization"]["method"]
                    )
                    if config["chemistry"]["optimization"]["method"]
                    in ["BFGS", "L-BFGS-B", "CG", "Newton-CG"]
                    else 0
                ),
                key="config_opt_method",
            )

            config["chemistry"]["optimization"]["fmax"] = st.number_input(
                "Force Max (eV/Å)",
                min_value=0.001,
                max_value=1.0,
                value=config["chemistry"]["optimization"]["fmax"],
                format="%.3f",
                key="config_fmax",
            )

            config["chemistry"]["optimization"]["steps"] = st.number_input(
                "Max Steps",
                min_value=1,
                max_value=1000,
                value=config["chemistry"]["optimization"]["steps"],
                key="config_steps",
            )

        with col4:
            st.write("**Calculators**")
            calc_options = ["mace_mp", "emt", "nwchem", "orca", "psi4", "tblite"]
            config["chemistry"]["calculators"]["default"] = st.selectbox(
                "Default Calculator",
                calc_options,
                index=(
                    calc_options.index(config["chemistry"]["calculators"]["default"])
                    if config["chemistry"]["calculators"]["default"] in calc_options
                    else 0
                ),
                key="config_calc_default",
            )

            config["chemistry"]["calculators"]["fallback"] = st.selectbox(
                "Fallback Calculator",
                calc_options,
                index=(
                    calc_options.index(config["chemistry"]["calculators"]["fallback"])
                    if config["chemistry"]["calculators"]["fallback"] in calc_options
                    else 1
                ),
                key="config_calc_fallback",
            )

    with tab2:
        st.subheader("API Settings")

        api_tabs = st.tabs(["OpenAI", "Anthropic", "Google", "Local"])

        with api_tabs[0]:
            config["api"]["openai"]["base_url"] = st.text_input(
                "Base URL",
                value=config["api"]["openai"]["base_url"],
                key="config_openai_url",
            )
            config["api"]["openai"]["timeout"] = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=300,
                value=config["api"]["openai"]["timeout"],
                key="config_openai_timeout",
            )

        with api_tabs[1]:
            config["api"]["anthropic"]["base_url"] = st.text_input(
                "Base URL",
                value=config["api"]["anthropic"]["base_url"],
                key="config_anthropic_url",
            )
            config["api"]["anthropic"]["timeout"] = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=300,
                value=config["api"]["anthropic"]["timeout"],
                key="config_anthropic_timeout",
            )

        with api_tabs[2]:
            config["api"]["google"]["base_url"] = st.text_input(
                "Base URL",
                value=config["api"]["google"]["base_url"],
                key="config_google_url",
            )
            config["api"]["google"]["timeout"] = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=300,
                value=config["api"]["google"]["timeout"],
                key="config_google_timeout",
            )

        with api_tabs[3]:
            config["api"]["local"]["base_url"] = st.text_input(
                "Base URL",
                value=config["api"]["local"]["base_url"],
                key="config_local_url",
            )
            config["api"]["local"]["timeout"] = st.number_input(
                "Timeout (seconds)",
                min_value=1,
                max_value=300,
                value=config["api"]["local"]["timeout"],
                key="config_local_timeout",
            )

    with tab3:
        st.subheader("Raw TOML Configuration")
        st.markdown(
            """
        Edit the raw TOML configuration directly. Be careful with syntax!
        """
        )

        try:
            config_text = toml.dumps(config)
        except Exception as e:
            st.error(f"Error serializing config: {e}")
            config_text = ""

        edited_config = st.text_area(
            "TOML Content", value=config_text, height=400, key="config_raw_toml"
        )

        if st.button("📝 Update from TOML", key="update_from_toml"):
            try:
                new_config = toml.loads(edited_config)
                st.session_state.config = new_config
                st.success("✅ Configuration updated from TOML!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Invalid TOML syntax: {e}")

    # Action buttons
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("💾 Save Configuration", type="primary"):
            if save_config(config):
                st.success("✅ Configuration saved to config.toml!")
            else:
                st.error("❌ Failed to save configuration")

    with col2:
        if st.button("🔄 Reload Configuration"):
            st.session_state.config = load_config()
            st.success("✅ Configuration reloaded!")
            st.rerun()

    with col3:
        if st.button("🗑️ Reset to Defaults"):
            st.session_state.config = get_default_config()
            st.success("✅ Configuration reset to defaults!")
            st.rerun()

    with col4:
        # Download button for config file
        try:
            config_download = toml.dumps(config)
            st.download_button(
                "📥 Download TOML",
                config_download,
                "config.toml",
                mime="application/toml",
            )
        except Exception as e:
            st.error(f"Error preparing download: {e}")

    # Configuration preview
    with st.expander("📊 Configuration Summary", expanded=False):
        st.write("**Current Configuration:**")
        st.write(f"- Model: {config['general']['model']}")
        st.write(f"- Workflow: {config['general']['workflow']}")
        st.write("- Temperature: 0.0 (optimized for tool calling)")
        st.write("- Max Tokens: 4000")
        st.write(
            f"- Default Calculator: {config['chemistry']['calculators']['default']}"
        )

        # Environment variables check
        st.write("**Environment Variables:**")
        api_keys = {
            "OPENAI_API_KEY": "OpenAI",
            "ANTHROPIC_API_KEY": "Anthropic",
            "GEMINI_API_KEY": "Google",
        }

        for env_var, provider in api_keys.items():
            if os.getenv(env_var):
                st.write(f"- {provider}: ✅ Set")
            else:
                st.write(f"- {provider}: ❌ Not set")

    # Stop execution here for Config page
    st.stop()

# -----------------------------------------------------------------------------
# Main Interface (only runs if not on About or Config page)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Main title & description
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Session-state init and configuration loading (MUST BE FIRST)
# -----------------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "last_config" not in st.session_state:
    st.session_state.last_config = None
if "config" not in st.session_state:
    st.session_state.config = load_config()

# Get configuration values
config = st.session_state.config
selected_model = config["general"]["model"]
selected_workflow = config["general"]["workflow"]
selected_output = config["general"]["output"]
structured_output = config["general"]["structured"]
generate_report = config["general"]["report"]
thread_id = config["general"]["thread"]

# -----------------------------------------------------------------------------
# Main Interface Header
# -----------------------------------------------------------------------------

st.title("🧪 ChemGraph ")

st.markdown(
    """
ChemGraph enables you to perform various **computational chemistry** tasks with
natural-language queries using AI agents.
"""
)

# Quick settings override
with st.sidebar.expander("🔧 Quick Settings"):
    st.write("Override settings for this session:")

    # Model override
    if st.checkbox("Override Model"):
        selected_model = st.selectbox(
            "Select Model",
            all_supported_models,
            index=(
                all_supported_models.index(selected_model)
                if selected_model in all_supported_models
                else 0
            ),
        )

    # Thread ID override
    if st.checkbox("Override Thread ID"):
        thread_id = st.number_input(
            "Thread ID", min_value=1, max_value=1000, value=thread_id
        )

    st.info("💡 To make permanent changes, use the Configuration page.")

# Reload config button
if st.sidebar.button("🔄 Reload Config"):
    st.session_state.config = load_config()
    st.success("✅ Configuration reloaded!")
    st.rerun()

# -----------------------------------------------------------------------------
# Agent status section
# -----------------------------------------------------------------------------
st.sidebar.header("🅒🅖 Agent Status")

if st.session_state.agent:
    st.sidebar.success("✅ Agents Ready")
    st.sidebar.info(f"🧠 Model: {selected_model}")
    st.sidebar.info(f"⚙️ Workflow: {selected_workflow}")
    st.sidebar.info(f"🔗 Thread ID: {thread_id}")
    st.sidebar.info(f"💬 Messages: {len(st.session_state.conversation_history)}")

    # Add a manual refresh button for troubleshooting
    if st.sidebar.button("🔄 Refresh Agents"):
        st.session_state.agent = None  # Force re-initialization
        st.rerun()
else:
    st.sidebar.error("❌ Agents Not Ready")
    st.sidebar.info("Agents will initialize automatically...")

# Configuration page link
st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Configuration**")
st.sidebar.markdown(
    "Use the Configuration page to modify settings, API endpoints, and chemistry parameters."
)
st.sidebar.markdown("Current config loaded from: `config.toml`")


from pathlib import Path
from datetime import datetime, timezone, timedelta

# -----------------------------------------------------------------------------
# Helper: check if IR spectrum file has changed within last minute
# -----------------------------------------------------------------------------


def changed_recently(path="ir_spectrum.png", window_seconds=60) -> bool:
    """
    Return True if `path` exists and was modified within the last `window_seconds`.
    """
    p = Path(path)
    if not p.exists():
        return False

    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    now = datetime.now(timezone.utc)
    return (now - mtime) <= timedelta(seconds=window_seconds)


# -----------------------------------------------------------------------------
# Helper: extract molecular structure from plain-text message
# -----------------------------------------------------------------------------
def find_html_filename(messages: list) -> Optional[str]:
    """
    Scan through *messages* in reverse order for the first occurrence of something
    that looks like an HTML file (e.g. 'report.html' or 'results/2025/plot.html').
    Returns the matched substring (path or bare filename) or `None` if nothing is found.

    Parameters
    ----------
    messages : list
        List of message objects to search through

    Returns
    -------
    str or None
        HTML filename/path if found, None otherwise

    Examples
    --------
    >>> messages = [{"content": "See docs in build/output/index.html"}, {"content": "No HTML"}]
    >>> find_html_filename(messages)
    'build/output/index.html'

    >>> find_html_filename([{"content": "No HTML here"}])
    None
    """
    pattern = r"[\w./-]+\.html\b"  # words / dots / slashes up to '.html'

    # Search through messages in reverse order (most recent first)
    for message in reversed(messages):
        # Extract content from different message formats
        content = ""
        if hasattr(message, "content"):
            content = getattr(message, "content", "")
        elif isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(message, str):
            content = message
        else:
            content = str(message)

        # Search for HTML pattern in this message content
        if content:
            match = re.search(pattern, content, flags=re.IGNORECASE)
            if match:
                return match.group(0)  # Return immediately when found

    return None  # No HTML filename found in any message


def extract_molecular_structure(message_content: str):
    """Return dict with keys atomic_numbers, positions if embedded in message."""
    if not message_content:
        return None

    # First try to parse as JSON (for structured output)
    try:
        # Check if the content is JSON with structure data
        if message_content.strip().startswith("{") and message_content.strip().endswith(
            "}"
        ):
            json_data = json.loads(message_content)

            # Look for structure data in various JSON formats
            structure_data = None
            if "answer" in json_data:
                structure_data = json_data["answer"]
            elif "numbers" in json_data and "positions" in json_data:
                structure_data = json_data
            elif "atomic_numbers" in json_data and "positions" in json_data:
                structure_data = json_data

            if (
                structure_data
                and "numbers" in structure_data
                and "positions" in structure_data
            ):
                return {
                    "atomic_numbers": structure_data["numbers"],
                    "positions": structure_data["positions"],
                }
            elif (
                structure_data
                and "atomic_numbers" in structure_data
                and "positions" in structure_data
            ):
                return {
                    "atomic_numbers": structure_data["atomic_numbers"],
                    "positions": structure_data["positions"],
                }
    except (json.JSONDecodeError, KeyError):
        pass

    # Then try to parse plain text format (original method)
    lines = message_content.splitlines()
    atomic_numbers, positions = None, None

    for i, line in enumerate(lines):
        if "Atomic Numbers" in line:
            try:
                numbers_str = line.split(":")[1].strip()
                atomic_numbers = ast.literal_eval(numbers_str)
            except Exception:
                pass
        elif "Positions" in line:
            positions = []
            for sub in lines[i + 1 :]:
                sub = sub.strip()
                if sub.startswith("- [") and sub.endswith("]"):
                    try:
                        positions.append(ast.literal_eval(sub[2:]))
                    except Exception:
                        pass
                elif not sub.startswith("-") and positions:
                    break

    if (
        isinstance(atomic_numbers, list)
        and isinstance(positions, list)
        and len(atomic_numbers) == len(positions)
    ):
        return {"atomic_numbers": atomic_numbers, "positions": positions}

    return None


# Helper: extract messages from result object
def extract_messages_from_result(result):
    """Extract messages from result object, handling different formats."""
    if isinstance(result, list):
        return result  # Already a list of messages
    elif isinstance(result, dict) and "messages" in result:
        messages = result["messages"]

        # For multi-agent workflows, also extract messages from worker_channel
        if "worker_channel" in result:
            worker_channel = result["worker_channel"]
            # Flatten all worker messages into the main messages list
            for worker_id, worker_messages in worker_channel.items():
                if isinstance(worker_messages, list):
                    messages.extend(worker_messages)

        return messages
    else:
        return [result]  # Treat as single message


# Helper: find structure data in messages
def find_structure_in_messages(messages):
    """Look through all messages to find structure data."""
    for message in messages:
        if hasattr(message, "content") or isinstance(message, dict):
            content = (
                getattr(message, "content", "")
                if hasattr(message, "content")
                else message.get("content", "")
            )
            structure = extract_molecular_structure(content)
            if structure:
                return structure
    return None


def is_infrared_requested(messages):
    """Look through all messages to find infrared data."""
    for message in messages:
        # Handle different message formats
        content = ""
        if hasattr(message, "content"):
            content = getattr(message, "content", "")
        elif isinstance(message, dict):
            content = message.get("content", "")
        elif isinstance(message, str):
            content = message
        else:
            content = str(message)

        if content and (("infrared" in content.lower()) or ("IR" in content)):
            return True


# Streamlit-specific wrapper for ASE functions
def create_ase_atoms_with_streamlit_error(atomic_numbers, positions):
    """Wrapper for create_ase_atoms that displays errors in Streamlit."""
    atoms = create_ase_atoms(atomic_numbers, positions)
    if atoms is None:
        st.error("Error creating ASE Atoms object")
    return atoms


# -----------------------------------------------------------------------------
# Display 3-D (or fallback) molecular structure
# -----------------------------------------------------------------------------
def display_molecular_structure(atomic_numbers, positions, title="Structure"):
    try:
        atoms = create_ase_atoms_with_streamlit_error(atomic_numbers, positions)
        if atoms is None:
            return False

        xyz_string = create_xyz_string(atomic_numbers, positions)
        if xyz_string is None:
            return False

        st.subheader(f"🧬 {title}")
        col1, col2 = st.columns([2, 1])

        # 3-D panel ------------------------------------------------------------
        with col1:
            if STMOL_AVAILABLE:
                style_options = ["ball_and_stick", "stick", "sphere", "wireframe"]
                selected_style = st.selectbox(
                    "Visualization Style", style_options, key=f"style_{uuid4().hex}"
                )

                # Create the 3D visualization using stmol directly
                try:
                    import py3Dmol

                    # Create py3Dmol viewer
                    view = py3Dmol.view(width=500, height=400)
                    view.addModel(xyz_string, "xyz")

                    if selected_style == "ball_and_stick":
                        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})
                    elif selected_style == "stick":
                        view.setStyle({"stick": {}})
                    elif selected_style == "sphere":
                        view.setStyle({"sphere": {}})
                    elif selected_style == "wireframe":
                        view.setStyle({"line": {}})
                    else:
                        view.setStyle({"stick": {}, "sphere": {"scale": 0.3}})

                    view.zoomTo()

                    # Use stmol.showmol with the py3Dmol view object
                    stmol.showmol(view, height=400, width=500)

                except Exception as viz_error:
                    st.error(f"3D visualization error: {viz_error}")
                    st.info("Falling back to table view...")
                    # Show fallback table
                    data = []
                    for idx, (num, pos) in enumerate(zip(atomic_numbers, positions), 1):
                        sym = (
                            chemical_symbols[num]
                            if num < len(chemical_symbols)
                            else f"X{num}"
                        )
                        data.append(
                            {
                                "Atom": idx,
                                "Element": sym,
                                "X": f"{pos[0]:.4f}",
                                "Y": f"{pos[1]:.4f}",
                                "Z": f"{pos[2]:.4f}",
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(data), height=350, use_container_width=True
                    )
            else:
                st.info("3-D viewer unavailable; showing raw XYZ and table.")

                # Show XYZ content
                with st.expander("📄 XYZ Format", expanded=True):
                    st.code(xyz_string, language="text")

                # Show structure table
                data = []
                for idx, (num, pos) in enumerate(zip(atomic_numbers, positions), 1):
                    sym = (
                        chemical_symbols[num]
                        if num < len(chemical_symbols)
                        else f"X{num}"
                    )
                    data.append(
                        {
                            "Atom": idx,
                            "Element": sym,
                            "X": f"{pos[0]:.4f}",
                            "Y": f"{pos[1]:.4f}",
                            "Z": f"{pos[2]:.4f}",
                        }
                    )
                st.dataframe(pd.DataFrame(data), height=350, use_container_width=True)

        # Info panel -----------------------------------------------------------
        with col2:
            st.markdown("**Structure Information**")
            st.write(f"- **Atoms:** {len(atoms)}")
            st.write(f"- **Formula:** {atoms.get_chemical_formula()}")

            # Composition
            composition = {}
            for atom in atoms:
                composition[atom.symbol] = composition.get(atom.symbol, 0) + 1
            st.write("**Composition:**")
            for elem, count in sorted(composition.items()):
                st.write(f"  • {elem}: {count}")

            # Total mass
            try:
                total_mass = atoms.get_masses().sum()
                st.write(f"**Total Mass:** {total_mass:.2f} amu")
            except:
                st.write("**Total Mass:** Not available")

            # Center of mass
            try:
                com = atoms.get_center_of_mass()
                st.write(f"**Center of Mass:**")
                st.write(f"  [{com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f}] Å")
            except:
                st.write("**Center of Mass:** Not available")

            # Additional properties
            with st.expander("🔬 Additional Properties"):
                try:
                    pos = atoms.positions
                    com = atoms.get_center_of_mass()
                    distances = np.linalg.norm(pos - com, axis=1)
                    st.write(f"**Max distance from COM:** {distances.max():.3f} Å")
                    st.write(f"**Min distance from COM:** {distances.min():.3f} Å")

                    cell = atoms.get_cell()
                    if np.any(cell.lengths()):  # any non-zero → periodic
                        st.write(f"**Cell lengths:** {cell.lengths()}")
                        st.write(f"**Cell angles:** {cell.angles()}")
                    else:
                        st.write("**Cell:** non-periodic")
                except Exception as prop_error:
                    st.write(f"Error calculating properties: {prop_error}")

            # Downloads
            st.write("**Download:**")
            st.download_button(
                "📄 XYZ File",
                xyz_string,
                f"{title.lower().replace(' ', '_')}.xyz",
                mime="chemical/x-xyz",
            )

            structure_json = json.dumps(
                {
                    "atomic_numbers": atomic_numbers,
                    "positions": positions,
                    "formula": atoms.get_chemical_formula(),
                    "symbols": atoms.get_chemical_symbols(),
                },
                indent=2,
            )
            st.download_button(
                "📋 JSON Data",
                structure_json,
                f"{title.lower().replace(' ', '_')}.json",
                mime="application/json",
            )

        return True
    except Exception as exc:
        st.error(f"Error displaying structure: {exc}")
        return False


# Function for IR spectrum rendering

import base64
import json
import base64
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np


# -----------------------------------------------------------------------------
# Agent initializer (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_agent(
    model_name,
    workflow_type,
    structured_output,
    return_option,
    generate_report,
    recursion_limit,
):
    try:
        from chemgraph.agent.llm_agent import ChemGraph

        return ChemGraph(
            model_name=model_name,
            workflow_type=workflow_type,
            generate_report=generate_report,
            return_option=return_option,
            recursion_limit=recursion_limit,
        )
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return None


# -----------------------------------------------------------------------------
# Auto-initialize agent when configuration changes
# -----------------------------------------------------------------------------
current_config = (
    selected_model,
    selected_workflow,
    structured_output,
    selected_output,
    generate_report,
    config["general"]["recursion_limit"],
)

if st.session_state.agent is None or st.session_state.last_config != current_config:

    with st.spinner("🚀 Initializing ChemGraph agents..."):
        st.session_state.agent = initialize_agent(
            selected_model,
            selected_workflow,
            structured_output,
            selected_output,
            generate_report,
            config["general"]["recursion_limit"],
        )
        st.session_state.last_config = current_config

        if st.session_state.agent:
            st.success("✅ ChemGraph agents ready!")
        else:
            st.error("❌ Agent initialization failed.")


# -----------------------------------------------------------------------------
# Main chat interface
# -----------------------------------------------------------------------------

# Conversation history display
if st.session_state.conversation_history:
    st.subheader("🗨️ Conversation History")
    for idx, entry in enumerate(st.session_state.conversation_history, 1):
        # User bubble
        st.markdown(
            f"""
<div style="background:#e3f2fd;padding:15px;border-radius:15px;margin:10px 0 0 50px;border:1px solid #2196f3;color:#000000;">
  <b style="color:#1976d2;">👤 You:</b><br><span style="color:#333333;">{entry["query"]}</span>
</div>""",
            unsafe_allow_html=True,
        )

        # Extract messages from the result
        messages = extract_messages_from_result(entry["result"])

        # Find the final AI response for display
        final_answer = ""
        for message in reversed(messages):
            # Handle different message formats
            if hasattr(message, "content") and hasattr(message, "type"):
                # LangChain message object
                if message.type == "ai" and message.content.strip():
                    # Skip if it's just JSON structure data
                    content = message.content.strip()
                    if not (
                        content.startswith("{")
                        and content.endswith("}")
                        and "numbers" in content
                    ):
                        final_answer = content
                        break
            elif isinstance(message, dict):
                # Dictionary message format
                if message.get("type") == "ai" and message.get("content", "").strip():
                    content = message["content"].strip()
                    if not (
                        content.startswith("{")
                        and content.endswith("}")
                        and "numbers" in content
                    ):
                        final_answer = content
                        break
            elif hasattr(message, "content"):
                # Generic message object with content
                content = getattr(message, "content", "").strip()
                if content and not (
                    content.startswith("{")
                    and content.endswith("}")
                    and "numbers" in content
                ):
                    final_answer = content
                    break

        # Display the AI response
        if final_answer:
            st.markdown(
                f"""
<div style="background:#f1f8e9;padding:15px;border-radius:15px;margin:10px 50px 0 0;border:1px solid #4caf50;color:#000000;">
  <b style="color:#388e3c;">🅒🅖 ChemGraph:</b><br><span style="color:#333333;">{final_answer.replace(chr(10), "<br>")}</span>
</div>""",
                unsafe_allow_html=True,
            )

        # Look for structure data across all messages
        structure = find_structure_in_messages(messages)
        if structure:
            display_molecular_structure(
                structure["atomic_numbers"],
                structure["positions"],
                title=f"Molecular Structure (Query {idx})",
            )
        else:
            # Also check the final answer text for structure data
            structure_from_text = extract_molecular_structure(final_answer)
            if structure_from_text:
                display_molecular_structure(
                    structure_from_text["atomic_numbers"],
                    structure_from_text["positions"],
                    title=f"Structure from Response {idx}",
                )
        html_filename = find_html_filename(messages)
        if html_filename:
            with st.expander(f"📊 Report", expanded=False):
                # st.subheader(" Generated Report")
                try:
                    with open(html_filename, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600, scrolling=True)
                except FileNotFoundError:
                    st.warning(f"HTML file '{html_filename}' not found")
                except Exception as e:
                    st.error(f"Error displaying HTML: {e}")

        # Check for embedded HTML plots/snippets in all messages

        if is_infrared_requested(messages):
            if changed_recently():
                with st.expander(f"🔍 IR Spectrum", expanded=True):
                    st.image("ir_spectrum.png")
            else:
                st.warning("IR spectrum not found.")

        # Optional debug information
        with st.expander(f"🔍 Verbose Info (Query {idx})", expanded=False):
            st.write(f"**Number of messages:** {len(messages)}")
            st.write(f"**Structure found:** {'Yes' if structure else 'No'}")

            # Show message types and content summaries
            for i, msg in enumerate(messages):
                if hasattr(msg, "type"):
                    msg_type = msg.type
                    content = msg.content
                    content_preview = (
                        (msg.content[:100] + "...")
                        if len(msg.content) > 100
                        else msg.content
                    )
                elif isinstance(msg, dict):
                    msg_type = msg.get("type", "unknown")
                    content = msg.get("content", "")
                    content_preview = (
                        (content[:100] + "...") if len(content) > 100 else content
                    )
                else:
                    msg_type = type(msg).__name__
                    content = getattr(msg, "content", str(msg)[:100])
                    content_preview = (
                        (content[:100] + "...") if len(content) > 100 else content
                    )

                st.write(f"  **Message {i+1}:** `{msg_type}` - {content}")

        st.markdown("---")

# -----------------------------------------------------------------------------
# New query input
# -----------------------------------------------------------------------------

with st.expander("💡 Example Queries"):
    st.markdown("**Based on your current configuration:**")
    st.markdown(f"- Model: {selected_model}")
    st.markdown(
        f"- Default Calculator: {config['chemistry']['calculators']['default']}"
    )
    st.markdown("- Temperature: 0.0 (optimized for tool calling)")

    examples = [
        "What is the SMILES string for caffeine?",
        f"Optimize the geometry of water molecule using {config['chemistry']['calculators']['default']}",
        "Calculate the single point energy of methane and show the structure",
        "Generate the molecular structure of aspirin and calculate its vibrational frequencies",
        "Compare the energy of different conformers of ethane",
        "What are the bond lengths in optimized CO2 molecule?",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            # Set the example text directly in the text area state
            st.session_state.query_input = ex
            st.rerun()

# Initialize query input if not exists
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

query = st.text_area(
    "Enter your computational chemistry query:",
    value=st.session_state.query_input,
    height=100,
    key="query_text_area",  # Different key to avoid conflicts
)

# Update session state with current text area value
if query != st.session_state.query_input:
    st.session_state.query_input = query

col_send, col_clear, col_refresh = st.columns([2, 1, 1])

send = col_send.button("🚀 Send", type="primary", use_container_width=True)
if col_clear.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.conversation_history.clear()
    # Clear the query input
    st.session_state.query_input = ""
    st.rerun()
if col_refresh.button("🔄 Refresh", use_container_width=True):
    st.rerun()

# -----------------------------------------------------------------------------
# Submit query
# -----------------------------------------------------------------------------
if send:
    if not st.session_state.agent:
        st.error("❌ Agent not ready. Please check configuration and try again.")
        if st.button("🔄 Try Again"):
            st.rerun()
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("ChemGraph agents working...", show_time=True):
            try:
                cfg = {"configurable": {"thread_id": thread_id}}
                result = st.session_state.agent.run(query.strip(), config=cfg)
                st.session_state.conversation_history.append(
                    {"query": query.strip(), "result": result, "thread_id": thread_id}
                )
                # Clear the input after successful processing
                st.session_state.query_input = ""
                st.success("✅ Done!")
                st.rerun()
            except Exception as exc:
                st.error(f"Processing error: {exc}")

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
### Quick Help

**Main Features:** Molecular optimization, vibrational frequencies, SMILES ↔ structure conversions, 3D visualization

📖 For detailed information, documentation, and links to research papers, visit the **About ChemGraph** page.
"""
)
