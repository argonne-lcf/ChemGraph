from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from chemgraph.tools.ase_tools import (
    run_ase,
    save_atomsdata_to_file,
    file_to_atomsdata,
)
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
)
from chemgraph.tools.generic_tools import calculator
from chemgraph.prompt.single_agent_prompt import (
    single_agent_prompt,
)
from chemgraph.utils.logging_config import setup_logger
from chemgraph.state.state import State

logger = setup_logger(__name__)


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
            file_to_atomsdata,
            smiles_to_atomsdata,
            run_ase,
            molecule_name_to_smiles,
            save_atomsdata_to_file,
            calculator,
        ]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}

def construct_mock_agent_graph(
    llm: ChatOpenAI,
    system_prompt: str = single_agent_prompt,
    tools: list = None,
):
    """Construct a geometry optimization graph.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use for the graph
    system_prompt : str, optional
        The system prompt to guide the LLM's behavior, by default single_agent_prompt
    tools: list, optional
        The list of tools for the main agent, by default None
    Returns
    -------
    StateGraph
        The constructed single agent graph
    """
    logger.info("Constructing mock agent graph")
    checkpointer = MemorySaver()
    if tools is None:
        tools = [
            file_to_atomsdata,
            smiles_to_atomsdata,
            run_ase,
            molecule_name_to_smiles,
            save_atomsdata_to_file,
            calculator,
        ]
    graph_builder = StateGraph(State)

    graph_builder.add_node(
        "ChemGraphAgent",
        lambda state: ChemGraphAgent(state, llm, system_prompt=system_prompt, tools=tools),
    )
    graph_builder.add_edge(START, "ChemGraphAgent")
    graph_builder.add_edge("ChemGraphAgent", END)

    graph = graph_builder.compile(checkpointer=checkpointer)
    logger.info("Mock agent graph construction completed")
    return graph
