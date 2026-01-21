from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_coordinate_file,
)

from chemgraph.tools.architector_tools import (
    visualize_molecule,
    image_to_connection_points,
    build_metal_complex
)
from chemgraph.utils.logging_config import setup_logger
from chemgraph.state.state import State

logger = setup_logger(__name__)

single_agent_prompt = ""

def route_tools(state: State):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to 'done'.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps

    Returns
    -------
    str
        Either 'tools' or 'done' based on the state conditions
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return "done"


def ChemGraphAgent(state: State, llm: ChatOpenAI, system_prompt: str, tools=None):
    """LLM node that processes messages and decides next actions.

    Parameters
    ----------
    state : State
        The current state containing messages and remaining steps
    llm : ChatOpenAI
        The language model to use for processing
    system_prompt : str
        The system prompt to guide the LLM's behavior
    tools : list, optional
        List of tools available to the agent, by default None

    Returns
    -------
    dict
        Updated state containing the LLM's response
    """

    # Load default tools if no tool is specified.
    if tools is None:
        tools = [
            molecule_name_to_smiles,
            smiles_to_coordinate_file,
            visualize_molecule,
            image_to_connection_points,
            build_metal_complex
        ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}

def construct_single_agent_architector_graph(
    llm: ChatOpenAI,
    system_prompt: str = "",
    tools: list = None,
):
    """Construct a geometry optimization graph.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use for the graph
    system_prompt : str, optional
        The system prompt to guide the LLM's behavior, by default single_agent_prompt
    structured_output : bool, optional
        Whether to use structured output, by default False
    formatter_prompt : str, optional
        The prompt to guide the LLM's formatting behavior, by default formatter_prompt
    generate_report: bool, optional
        Whether to generate a report, by default False
    report_prompt: str, optional
        The prompt to guide the LLM's report generation behavior, by default report_prompt
    tool: list, optional
        The list of tools for the main agent, by default None
    Returns
    -------
    StateGraph
        The constructed single agent graph
    """
    try:
        logger.info("Constructing single agent graph")
        checkpointer = MemorySaver()
        if tools is None:
            tools = [
                molecule_name_to_smiles,
                smiles_to_coordinate_file,
                visualize_molecule,
                image_to_connection_points,
                build_metal_complex
            ]
        tool_node = ToolNode(tools=tools)
        graph_builder = StateGraph(State)

        graph_builder.add_node(
            "ChemGraphAgent",
            lambda state: ChemGraphAgent(state, llm, system_prompt=system_prompt, tools=tools),
        )
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_edge(START, "ChemGraphAgent")
        graph_builder.add_conditional_edges(
            "ChemGraphAgent",
            route_tools,
            {"tools": "tools", "done": END},
        )
        graph_builder.add_edge("tools", "ChemGraphAgent")
        graph_builder.add_edge("ChemGraphAgent", END)

        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Graph construction completed")
        return graph
    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
