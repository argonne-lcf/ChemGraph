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
from io import StringIO
from uuid import uuid4
import re
from typing import Optional

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

# -----------------------------------------------------------------------------
# Optional 3-D viewer - stmol + py3Dmol
# -----------------------------------------------------------------------------
try:
    import stmol

    # Check if stmol works by testing a simple import
    from stmol import showmol

    STMOL_AVAILABLE = True
    st.success("✅ 3D molecular visualization available!")
except ImportError as e:
    STMOL_AVAILABLE = False
    st.warning("⚠️ **stmol** not available – falling back to text/table view.")
    st.info("To enable 3D visualization, install with: `pip install stmol`")

# -----------------------------------------------------------------------------
# Main title & description
# -----------------------------------------------------------------------------
st.title("🧪 ChemGraph — AI Agents for Computational Chemistry")

st.markdown(
    """
ChemGraph enables you to perform various **computational chemistry** tasks using
natural-language queries.
"""
)

# -----------------------------------------------------------------------------
# Sidebar – configuration
# -----------------------------------------------------------------------------
st.sidebar.header("Configuration")

model_options = ["gpt-4o-mini", "gpt-4o", "claude-3-5-haiku-20241022"]
selected_model = st.sidebar.selectbox("Select LLM Model", model_options, index=0)

workflow_options = ["single_agent", "multi_agent", "python_repl", "graspa"]
selected_workflow = st.sidebar.selectbox("Workflow Type", workflow_options, index=0)

output_options = ["state", "last_message"]
selected_output = st.sidebar.selectbox("Return Option", output_options, index=0)

structured_output = st.sidebar.checkbox("Structured Output", value=False)
generate_report = st.sidebar.checkbox("Generate Report", value=False)

thread_id = st.sidebar.number_input(
    "Thread ID", min_value=1, max_value=1000, value=1, help="Conversation thread"
)

# -----------------------------------------------------------------------------
# Session-state init
# -----------------------------------------------------------------------------
if "agent" not in st.session_state:
    st.session_state.agent = None
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# -----------------------------------------------------------------------------
# Agent status section
# -----------------------------------------------------------------------------
st.sidebar.header("🅒🅖 Agent Status")

if st.session_state.agent:
    st.sidebar.success("✅ Agent Ready")
    st.sidebar.info(f"🧠 Model: {selected_model}")
    st.sidebar.info(f"⚙️ Workflow: {selected_workflow}")
    st.sidebar.info(f"🔗 Thread ID: {thread_id}")
    st.sidebar.info(f"💬 Messages: {len(st.session_state.conversation_history)}")

    # if st.sidebar.button("📊 Visualize Workflow"):

    #     st.sidebar.image("📈 Workflow visualization placeholder")
else:
    st.sidebar.warning("⚠️ Agent Not Initialized")


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
        return result["messages"]  # Extract from messages key
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


# NOTE: ASE helper functions moved to src/chemgraph/tools/ase_tools.py
# Functions now imported: create_ase_atoms, create_xyz_string, extract_ase_atoms_from_tool_result


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


# -----------------------------------------------------------------------------
# Agent initializer (cached)
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_agent(model_name, workflow_type, structured_output, return_option):
    try:
        from chemgraph.agent.llm_agent import ChemGraph

        return ChemGraph(
            model_name=model_name,
            workflow_type=workflow_type,
            generate_report=True,
            return_option="state",
            recursion_limit=20,
        )
    except Exception as exc:
        st.error(f"Failed to initialize agent: {exc}")
        return None


# -----------------------------------------------------------------------------
# Structure Test Section (after function definitions)
# -----------------------------------------------------------------------------
st.sidebar.header("🧪 Structure Test")
if st.sidebar.button("Test Structure Parsing"):
    # Test with the sample data from the user
    test_json = '{"answer":{"numbers":[8,1,1],"positions":[[0.0,0.0,0.0],[0.0,0.0,0.96],[0.0,0.76,-0.32]],"cell":null,"pbc":null}}'

    st.sidebar.write("**Testing with sample JSON:**")
    structure = extract_molecular_structure(test_json)

    if structure:
        st.sidebar.success("✅ Structure parsing works!")
        st.sidebar.write(f"Found {len(structure['atomic_numbers'])} atoms")

        # Display the test structure
        display_molecular_structure(
            structure["atomic_numbers"],
            structure["positions"],
            title="Test Structure (Water Molecule)",
        )
    else:
        st.sidebar.error("❌ Structure parsing failed")


# -----------------------------------------------------------------------------
# Sidebar button: initialize agent
# -----------------------------------------------------------------------------
if st.sidebar.button("Initialize Agent"):
    with st.spinner("Initializing ChemGraph…"):
        st.session_state.agent = initialize_agent(
            selected_model, selected_workflow, structured_output, selected_output
        )
        if st.session_state.agent:
            st.sidebar.success("Agent initialized!")
        else:
            st.sidebar.error("Agent initialization failed.")

# -----------------------------------------------------------------------------
# Main chat interface
# -----------------------------------------------------------------------------
st.header("💬 Chat Interface")

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
st.subheader("💭 Ask a New Question")

with st.expander("💡 Example Queries"):
    examples = [
        "What is the SMILES string for caffeine?",
        "Optimize the geometry of water molecule using DFT",
        "Calculate the single point energy of methane and show the structure",
    ]
    for ex in examples:
        if st.button(ex, key=f"ex_{ex}"):
            st.session_state.selected_example = ex
            st.rerun()

# Use the selected example if available, otherwise empty
initial_value = st.session_state.get("selected_example", "")
if "selected_example" in st.session_state:
    del st.session_state.selected_example  # Clear it after using

query = st.text_area(
    "Enter your computational chemistry question:",
    value=initial_value,
    height=100,
    key="query_input",
)

col_send, col_clear, col_refresh = st.columns([2, 1, 1])

send = col_send.button("🚀 Send", type="primary", use_container_width=True)
if col_clear.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.conversation_history.clear()
    # Clear the query input by using a different key approach
    if "query_input" in st.session_state:
        del st.session_state.query_input
    st.rerun()
if col_refresh.button("🔄 Refresh", use_container_width=True):
    st.rerun()

# -----------------------------------------------------------------------------
# Submit query
# -----------------------------------------------------------------------------
if send:
    if not st.session_state.agent:
        st.error("Please initialize the agent from the sidebar.")
    elif not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Agent thinking…"):
            try:
                cfg = {"configurable": {"thread_id": thread_id}}
                result = st.session_state.agent.run(query.strip(), config=cfg)
                st.session_state.conversation_history.append(
                    {"query": query.strip(), "result": result, "thread_id": thread_id}
                )
                # Clear the input after successful processing
                # Check if there's an HTML file to display

                if "query_input" in st.session_state:
                    del st.session_state.query_input
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
### About ChemGraph  

Natural-language interface to computational chemistry tools.

**Features**

* Molecular geometry optimisation  
* Vibrational-frequency calculations  
* SMILES ↔ structure conversions  
* 3-D interactive visualisation (via *stmol*)  
* Download structures as XYZ or JSON
"""
)
