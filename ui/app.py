import streamlit as st
import json
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

# Agent Status in sidebar
st.sidebar.header("🤖 Agent Status")

if st.session_state.agent:
    st.sidebar.success("✅ Agent Ready")
    st.sidebar.info(f"🧠 Model: {selected_model}")
    st.sidebar.info(f"⚙️ Workflow: {selected_workflow}")
    st.sidebar.info(f"🔗 Thread ID: {thread_id}")
    st.sidebar.info(f"💬 Messages: {len(st.session_state.conversation_history)}")

    # Visualize workflow button
    if st.sidebar.button("📊 Visualize Workflow"):
        try:
            st.sidebar.info("📈 Workflow visualization would be displayed here")
        except Exception as e:
            st.sidebar.error(f"Could not visualize workflow: {str(e)}")
else:
    st.sidebar.warning("⚠️ Agent Not Initialized")
    st.sidebar.info("👆 Use the configuration above to initialize the agent")


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
st.header("💬 Chat Interface")

# Chat history display
if st.session_state.conversation_history:
    st.subheader("🗨️ Conversation History")

    # Display all conversations in chronological order
    for i, entry in enumerate(st.session_state.conversation_history):
        with st.container():
            # User message
            st.markdown(
                f"""
            <div style="background-color: #e3f2fd; padding: 15px; border-radius: 15px; margin: 10px 0; margin-left: 50px; border: 1px solid #2196f3;">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span style="font-size: 18px; margin-right: 8px;">👤</span>
                    <strong style="color: #1976d2;">You</strong>
                </div>
                <p style="margin: 0; font-size: 16px; color: #333;">{entry["query"]}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Agent response
            result = entry["result"]

            # Handle both single message and list of messages
            if isinstance(result, list):
                messages = result
            else:
                # Single message case - wrap in list for consistent processing
                messages = [result]

            if messages:
                # Find the final answer - look for the last AI message without tool calls
                final_answer = None
                for message in reversed(messages):
                    if (
                        type(message).__name__ == "AIMessage"
                        and hasattr(message, 'content')
                        and message.content
                        and message.content.strip()
                        and not (hasattr(message, 'tool_calls') and message.tool_calls)
                    ):
                        final_answer = message.content.strip()
                        break

                # If no final answer found, look for any AI message with content
                if not final_answer:
                    for message in reversed(messages):
                        if (
                            type(message).__name__ == "AIMessage"
                            and hasattr(message, 'content')
                            and message.content
                            and message.content.strip()
                        ):
                            final_answer = message.content.strip()
                            break

                # Display the response
                if final_answer:
                    st.markdown(
                        f"""
                    <div style="background-color: #f1f8e9; padding: 15px; border-radius: 15px; margin: 10px 0; margin-right: 50px; border: 1px solid #4caf50;">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 18px; margin-right: 8px;">🤖</span>
                            <strong style="color: #388e3c;">CompChemAgent</strong>
                        </div>
                        <div style="font-size: 16px; line-height: 1.6; color: #333;">
                            {final_answer.replace('\n', '<br>')}
                        </div>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    # Show processing status if no final answer
                    st.markdown(
                        """
                    <div style="background-color: #fff3e0; padding: 15px; border-radius: 15px; margin: 10px 0; margin-right: 50px; border: 1px solid #ff9800;">
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span style="font-size: 18px; margin-right: 8px;">🔄</span>
                            <strong style="color: #f57c00;">Processing...</strong>
                        </div>
                        <p style="margin: 0; font-size: 16px; color: #333;">Task completed. Check detailed results below.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Show detailed processing in expandable section
                with st.expander(
                    f"🔍 View detailed processing for query {i + 1}", expanded=False
                ):
                    # Show debug info for this conversation
                    st.write("**Debug Info:**")
                    st.write(f"- Result type: {type(entry['result']).__name__}")
                    st.write(f"- Number of messages: {len(messages)}")
                    st.write(
                        f"- Is single message: {not isinstance(entry['result'], list)}"
                    )

                    # Show all messages with their types and content preview
                    for j, msg in enumerate(messages):
                        msg_type = type(msg).__name__
                        has_content = hasattr(msg, 'content') and bool(
                            getattr(msg, 'content', '')
                        )
                        content_preview = ""
                        if has_content:
                            content = getattr(msg, 'content', '')
                            content_preview = (
                                f" - Content: {content[:100]}..."
                                if len(content) > 100
                                else f" - Content: {content}"
                            )

                        has_tool_calls = hasattr(msg, 'tool_calls') and bool(
                            getattr(msg, 'tool_calls', [])
                        )
                        tool_info = " - Has tool calls" if has_tool_calls else ""

                        st.write(f"  {j + 1}. {msg_type}{content_preview}{tool_info}")

                    # Count different message types
                    human_msgs = sum(
                        1 for msg in messages if type(msg).__name__ == "HumanMessage"
                    )
                    ai_msgs = sum(
                        1 for msg in messages if type(msg).__name__ == "AIMessage"
                    )
                    tool_msgs = sum(
                        1 for msg in messages if type(msg).__name__ == "ToolMessage"
                    )

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🤖 AI Messages", ai_msgs)
                    with col2:
                        st.metric("🔧 Tool Calls", tool_msgs)
                    with col3:
                        st.metric("📊 Total Steps", len(messages))

                    # Show all processing steps
                    for j, message in enumerate(messages):
                        message_type = type(message).__name__
                        message_content = getattr(message, 'content', '')

                        if (
                            message_type == "AIMessage"
                            and hasattr(message, 'tool_calls')
                            and message.tool_calls
                        ):
                            st.info(f"🔧 **Step {j + 1}**: Agent used tools")
                            for tool_call in message.tool_calls:
                                tool_name = tool_call.get('name', 'Unknown tool')
                                st.code(f"Tool: {tool_name}")

                        elif message_type == "ToolMessage" and message_content:
                            tool_name = getattr(message, 'name', 'Unknown')
                            st.success(
                                f"✅ **Step {j + 1}**: Tool result from {tool_name}"
                            )

                            try:
                                tool_result = json.loads(message_content)
                                if 'error' in tool_result:
                                    st.error(f"Error: {tool_result['error']}")
                                else:
                                    # Show key results
                                    if (
                                        'numbers' in tool_result
                                        and 'positions' in tool_result
                                    ):
                                        st.write(
                                            f"🧬 Found molecular structure with {len(tool_result['numbers'])} atoms"
                                        )
                                    elif 'vibrational_frequencies' in tool_result:
                                        if tool_result.get('success', False):
                                            st.write(
                                                "🎯 Calculation completed successfully"
                                            )
                                        else:
                                            st.write("⚠️ Calculation had issues")

                                    with st.expander("Raw data", expanded=False):
                                        st.json(tool_result)
                            except json.JSONDecodeError:
                                st.code(
                                    message_content[:200] + "..."
                                    if len(message_content) > 200
                                    else message_content
                                )

                        elif message_type == "AIMessage" and message_content:
                            st.info(f"🤖 **Step {j + 1}**: AI Response")
                            st.write(
                                message_content[:200] + "..."
                                if len(message_content) > 200
                                else message_content
                            )

            else:
                # Handle empty results
                st.markdown(
                    """
                <div style="background-color: #ffebee; padding: 15px; border-radius: 15px; margin: 10px 0; margin-right: 50px; border: 1px solid #f44336;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 18px; margin-right: 8px;">⚠️</span>
                        <strong style="color: #d32f2f;">No Response</strong>
                    </div>
                    <p style="margin: 0; font-size: 16px; color: #333;">No messages found in result.</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown("---")

# New query input at the bottom
st.subheader("💭 Ask a New Question")

# Example queries
with st.expander("💡 Example Queries", expanded=False):
    example_queries = [
        "What is the SMILES string for caffeine?",
        "What are the vibrational frequencies of CO2 using B3LYP/3-21G?",
        "Optimize the geometry of water molecule using DFT",
        "Calculate the single point energy of methane",
        "What is the molecular formula of benzene?",
        "What is the bond length of H2O?",
        "Calculate the dipole moment of ammonia",
        "What are the atomic charges in methanol?",
    ]

    for i, example in enumerate(example_queries):
        if st.button(f"📝 {example}", key=f"example_{i}"):
            st.session_state.selected_example = example
            st.session_state.auto_submit = True  # Flag to auto-submit
            st.rerun()

# Initialize session state
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = ""
if 'auto_submit' not in st.session_state:
    st.session_state.auto_submit = False

# Query input
query = st.text_area(
    "Enter your computational chemistry question:",
    value=st.session_state.selected_example,
    height=100,
    help="Ask questions about molecular properties, calculations, or optimizations",
    key="query_input",
)

col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    submit_button = st.button(
        "🚀 Send Message", type="primary", use_container_width=True
    )

with col2:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.conversation_history = []
        st.session_state.selected_example = ""
        st.rerun()

with col3:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()

# Auto-submit logic for selected examples or manual submit
if submit_button or st.session_state.auto_submit:
    if not st.session_state.agent:
        st.error("❌ Please initialize the agent first using the sidebar.")
        st.session_state.auto_submit = False
    elif not query or not query.strip():
        st.warning("⚠️ Please enter a question.")
        st.session_state.auto_submit = False
    else:
        # Use the actual query text
        query_text = query.strip()
        with st.spinner("🤔 Agent is thinking..."):
            try:
                # Run the agent
                config = {"configurable": {"thread_id": thread_id}}
                result = st.session_state.agent.run(query_text, config=config)

                # Store in conversation history
                st.session_state.conversation_history.append(
                    {"query": query_text, "result": result, "thread_id": thread_id}
                )

                # Clear states after successful submission
                st.session_state.selected_example = ""
                st.session_state.auto_submit = False

                st.success("✅ Response received!")
                st.rerun()  # Refresh to show the new message

            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")
                st.session_state.auto_submit = False

# Debug section (can be removed later)
if st.session_state.conversation_history:
    with st.expander("🔧 Debug Information", expanded=False):
        latest = st.session_state.conversation_history[-1]
        st.write("**Latest conversation data:**")
        st.json(
            {
                "query": latest["query"],
                "result_type": type(latest["result"]).__name__,
                "result_length": len(latest["result"])
                if isinstance(latest["result"], list)
                else "Not a list",
                "thread_id": latest.get("thread_id", "N/A"),
            }
        )
        if isinstance(latest["result"], list):
            st.write("**Message types in result:**")
            for i, msg in enumerate(latest["result"]):
                st.write(
                    f"- Message {i}: {type(msg).__name__} - Has content: {hasattr(msg, 'content') and bool(getattr(msg, 'content', ''))}"
                )

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
