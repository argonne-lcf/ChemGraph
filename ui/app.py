import streamlit as st
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set page configuration
st.set_page_config(
    page_title="CompChemAgent UI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main title
st.title("🧪 CompChemAgent - Computational Chemistry Assistant")

st.markdown("""
Welcome to the Computational Chemistry Agent interface! This tool allows you to perform various 
computational chemistry tasks using natural language queries.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Model selection
model_options = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo']
selected_model = st.sidebar.selectbox(
    "Select LLM Model",
    model_options,
    index=0,
    help="Choose the language model for the agent",
)

# Workflow type selection
workflow_options = ['single_agent_ase']
selected_workflow = st.sidebar.selectbox(
    "Workflow Type",
    workflow_options,
    index=0,
    help="Select the type of computational workflow",
)

# Output options
output_options = ['last_message', 'state']
selected_output = st.sidebar.selectbox(
    "Return Option",
    output_options,
    index=0,
    help="Choose what to return from the agent",
)

# Structured output toggle
structured_output = st.sidebar.checkbox(
    "Structured Output", value=False, help="Enable structured output format"
)

# Thread ID for conversation state
thread_id = st.sidebar.number_input(
    "Thread ID",
    min_value=1,
    max_value=1000,
    value=1,
    help="Unique identifier for conversation thread",
)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []


# Function to initialize the agent
@st.cache_resource
def initialize_agent(model_name, workflow_type, structured_output, return_option):
    try:
        from comp_chem_agent.agent.llm_graph import llm_graph

        agent = llm_graph(
            model_name=model_name,
            workflow_type=workflow_type,
            structured_output=structured_output,
            return_option=return_option,
        )
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


# Initialize agent button
if st.sidebar.button("Initialize Agent"):
    with st.spinner("Initializing CompChemAgent..."):
        st.session_state.agent = initialize_agent(
            selected_model, selected_workflow, structured_output, selected_output
        )
        if st.session_state.agent:
            st.sidebar.success("Agent initialized successfully!")
        else:
            st.sidebar.error("Failed to initialize agent")

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.header("Query Interface")

    # Example queries
    st.subheader("Example Queries")
    example_queries = [
        "What is the SMILES string for caffeine?",
        "What are the vibrational frequencies of CO2 using B3LYP/3-21G?",
        "Optimize the geometry of water molecule using DFT",
        "Calculate the single point energy of methane",
        "What is the molecular formula of benzene?",
    ]

    selected_example = st.selectbox(
        "Choose an example query:",
        [""] + example_queries,
        help="Select a pre-defined query or enter your own below",
    )

    # Query input
    if selected_example:
        query = st.text_area(
            "Enter your computational chemistry query:",
            value=selected_example,
            height=100,
            help="Ask questions about molecular properties, calculations, or optimizations",
        )
    else:
        query = st.text_area(
            "Enter your computational chemistry query:",
            height=100,
            help="Ask questions about molecular properties, calculations, or optimizations",
        )

    # Submit button
    if st.button("Submit Query", type="primary"):
        if not st.session_state.agent:
            st.error("Please initialize the agent first using the sidebar.")
        elif not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Processing your query..."):
                try:
                    # Run the agent
                    config = {"configurable": {"thread_id": thread_id}}
                    result = st.session_state.agent.run(query, config=config)

                    # Store in conversation history
                    st.session_state.conversation_history.append(
                        {"query": query, "result": result, "thread_id": thread_id}
                    )

                    st.success("Query processed successfully!")

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")

with col2:
    st.header("Agent Status")

    if st.session_state.agent:
        st.success("✅ Agent Ready")
        st.info(f"Model: {selected_model}")
        st.info(f"Workflow: {selected_workflow}")
        st.info(f"Thread ID: {thread_id}")

        # Visualize workflow button
        if st.button("Visualize Workflow"):
            try:
                # This would show the workflow graph
                st.info("Workflow visualization would be displayed here")
                # Note: The actual visualization might need special handling in Streamlit
            except Exception as e:
                st.error(f"Could not visualize workflow: {str(e)}")
    else:
        st.warning("⚠️ Agent Not Initialized")
        st.info("Use the sidebar to configure and initialize the agent")

# Results section
if st.session_state.conversation_history:
    st.header("Results")

    # Show latest result
    latest = st.session_state.conversation_history[-1]

    with st.expander("Latest Result", expanded=True):
        st.subheader("Query:")
        st.write(latest["query"])

        st.subheader("Response:")
        if isinstance(latest["result"], list):
            # Display conversation messages
            for i, message in enumerate(latest["result"]):
                if hasattr(message, 'content') and message.content:
                    message_type = type(message).__name__
                    st.write(f"**{message_type}:** {message.content}")
        else:
            st.write(latest["result"])

    # Show conversation history
    if len(st.session_state.conversation_history) > 1:
        st.subheader("Conversation History")

        for i, entry in enumerate(reversed(st.session_state.conversation_history[:-1])):
            with st.expander(
                f"Query {len(st.session_state.conversation_history) - i - 1}: {entry['query'][:50]}..."
            ):
                st.write("**Query:**", entry["query"])
                st.write("**Thread ID:**", entry["thread_id"])

                if isinstance(entry["result"], list):
                    st.write("**Messages:**")
                    for message in entry["result"]:
                        if hasattr(message, 'content') and message.content:
                            message_type = type(message).__name__
                            st.write(f"- {message_type}: {message.content}")
                else:
                    st.write("**Result:**", entry["result"])

# Clear history button
if st.session_state.conversation_history:
    if st.button("Clear History"):
        st.session_state.conversation_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
### About CompChemAgent
This application provides a natural language interface to computational chemistry tools. 
You can ask questions about molecular properties, request calculations, and perform various 
computational chemistry tasks using simple English queries.

**Features:**
- SMILES string lookup and conversion
- Molecular geometry optimization
- Vibrational frequency calculations
- Single point energy calculations
- Integration with various quantum chemistry packages

**Note:** Some calculations may require specific software packages to be installed on the system.
""")
