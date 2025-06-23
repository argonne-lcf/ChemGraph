from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
import json
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from chemgraph.tools.generic_tools import repl_tool
from chemgraph.tools.generic_tools import calculator
from chemgraph.prompt.single_agent_prompt import single_agent_prompt
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class State(TypedDict):
    """Type definition for the state dictionary used in the graph.

    Attributes
    ----------
    messages : list
        List of messages in the conversation, annotated with add_messages
    """

    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that executes tools requested in the last AIMessage.

    This class processes tool calls from AI messages and executes the corresponding
    tools, handling their results and any potential errors.

    Parameters
    ----------
    tools : list
        List of tool objects that can be called by the node

    Attributes
    ----------
    tools_by_name : dict
        Dictionary mapping tool names to their corresponding tool objects
    """

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: State) -> State:
        """Execute tools requested in the last message.

        Parameters
        ----------
        inputs : State
            The current state containing messages

        Returns
        -------
        State
            Updated state containing tool execution results

        Raises
        ------
        ValueError
            If no message is found in the input state
        """
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name")
                if not tool_name or tool_name not in self.tools_by_name:
                    raise ValueError(f"Invalid tool name: {tool_name}")

                tool_result = self.tools_by_name[tool_name].invoke(tool_call.get("args", {}))

                # Handle different types of tool results
                result_content = (
                    tool_result.dict()
                    if hasattr(tool_result, "dict")
                    else (tool_result if isinstance(tool_result, dict) else str(tool_result))
                )

                outputs.append(
                    ToolMessage(
                        content=json.dumps(result_content),
                        name=tool_name,
                        tool_call_id=tool_call.get("id", ""),
                    )
                )

            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}),
                        name=tool_name if tool_name else "unknown_tool",
                        tool_call_id=tool_call.get("id", ""),
                    )
                )
        return {"messages": outputs}


def route_tools(state: State):
    """Route to the 'tools' node if the last message has tool calls; otherwise, route to END.

    Parameters
    ----------
    state : State
        The current state containing messages

    Returns
    -------
    str
        Either 'tools' or END based on the presence of tool calls

    Raises
    ------
    ValueError
        If no messages are found in the input state
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def CompChemAgent(state: State, llm: ChatOpenAI, system_prompt=single_agent_prompt, tools=None):
    """LLM node that processes messages and decides next actions.

    Parameters
    ----------
    state : State
        The current state containing messages
    llm : ChatOpenAI
        The language model to use for processing
    system_prompt : str, optional
        The system prompt to guide the LLM's behavior,
        by default single_agent_prompt
    tools : list, optional
        List of tools available to the agent, by default None

    Returns
    -------
    dict
        Updated state containing the LLM's response
    """
    if tools is None:
        tools = []
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def construct_relp_graph(llm: ChatOpenAI, system_prompt=single_agent_prompt):
    """Construct a graph for REPL-based Python execution workflow.

    This function creates a state graph that implements a workflow for executing
    Python code through a REPL interface, using LLM agents and tools.

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use in the workflow
    system_prompt : str, optional
        The system prompt to guide the LLM's behavior,
        by default single_agent_prompt

    Returns
    -------
    StateGraph
        A compiled state graph implementing the REPL workflow

    Raises
    ------
    Exception
        If there is an error during graph construction
    """
    try:
        logger.info("Constructing geometry optimization graph")
        checkpointer = MemorySaver()
        tools = [
            repl_tool,
            calculator,
        ]
        tool_node = BasicToolNode(tools=tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node(
            "CompChemAgent",
            lambda state: CompChemAgent(state, llm, system_prompt=system_prompt, tools=tools),
        )
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_conditional_edges(
            "CompChemAgent",
            route_tools,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "CompChemAgent")
        graph_builder.add_edge(START, "CompChemAgent")
        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Graph construction completed")
        return graph
    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
