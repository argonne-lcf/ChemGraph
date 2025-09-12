!!! note
      ChemGraph includes a **Streamlit web interface** that provides an intuitive, chat-based UI for interacting with computational chemistry agents. The interface supports 3D molecular visualization, conversation history, and easy access to various ChemGraph workflows.

### Features

- **🧪 Interactive Chat Interface**: Natural language queries for computational chemistry tasks
- **🧬 3D Molecular Visualization**: Interactive molecular structure display using `stmol` and `py3Dmol`
- **📊 Report Integration**: Embedded HTML reports from computational calculations
- **💾 Data Export**: Download molecular structures as XYZ or JSON files
- **🔧 Multiple Workflows**: Support for single-agent, multi-agent, Python REPL, and gRASPA workflows
- **🎨 Modern UI**: Clean, responsive interface with conversation bubbles and molecular properties display

### Installation Requirements

The Streamlit UI dependencies are included by default when you install ChemGraph:

```bash
# Install ChemGraph (includes UI dependencies)
pip install -e .
```

**Alternative Installation Options:**
```bash
# Install only UI dependencies separately (if needed)
pip install -e ".[ui]"

# Install with UMA support (separate environment recommended)
pip install -e ".[uma]"
```

### Running the Streamlit Interface

1. **Set up your API keys** (same as for notebooks):
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export ANTHROPIC_API_KEY="your_anthropic_api_key_here"
   ```

2. **Launch the Streamlit app**:
   ```bash
   streamlit run ui/app.py
   ```

3. **Access the interface**: Open your browser to `http://localhost:8501`

### Using the Interface

#### Configuration
- **Model Selection**: Choose from GPT-4o, GPT-4o-mini, or Claude models
- **Workflow Type**: Select single-agent, multi-agent, Python REPL, or gRASPA workflows


#### Interaction
1. **Initialize Agent**: Click "Initialize Agent" in the sidebar to set up your ChemGraph instance
2. **Ask Questions**: Use the text area to enter computational chemistry queries
3. **View Results**: See responses in chat bubbles with automatic structure detection
4. **3D Visualization**: When molecular structures are detected, they're automatically displayed in 3D
5. **Download Data**: Export structures and calculation results directly from the interface

#### Example Queries
- "What is the SMILES string for caffeine?"
- "Optimize the geometry of water molecule using DFT"
- "Calculate the single point energy of methane and show the structure"
- "Generate the structure of aspirin and calculate its vibrational frequencies"

#### Molecular Visualization
The interface automatically detects molecular structure data in agent responses and provides:
- **Interactive 3D Models**: Multiple visualization styles (ball & stick, sphere, stick, wireframe)
- **Structure Information**: Chemical formula, composition, mass, center of mass
- **Export Options**: Download as XYZ files or JSON data
- **Fallback Display**: Table view when 3D visualization is unavailable

#### Conversation Management
- **History Display**: All queries and responses are preserved in conversation bubbles
- **Structure Detection**: Molecular structures are automatically extracted and visualized
- **Report Integration**: HTML reports from calculations are embedded directly in the interface
- **Debug Information**: Expandable sections show detailed message processing information

### Troubleshooting

**3D Visualization Issues:**
- Ensure `stmol` is installed: `pip install stmol`
- If 3D display fails, the interface falls back to table/text display
- Check browser compatibility for WebGL support

**Agent Initialization:**
- Verify API keys are set correctly
- Check that ChemGraph package is installed: `pip install -e .`
- Ensure all dependencies are available in your environment

**Performance:**
- For large molecular systems, visualization may take longer to load
- Use the refresh button if the interface becomes unresponsive
- Clear conversation history to improve performance with many queries
