from langgraph.graph import StateGraph, MessagesState, START, END
from comp_chem_agent.models.xtb import XTBSimulationInput
from comp_chem_agent.tools.xtb_tools import run_xtb_calculation
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
from typing import Annotated, Literal
from typing_extensions import TypedDict
import json
from langchain_openai import ChatOpenAI

class State(TypedDict):
    messages: Annotated[list, add_messages]

class run_xtb:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: State) -> State:
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result.dict()),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}
    
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
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

def setup_simulation(llm_with_tools: ChatOpenAI, state: State):
    """First node: Creates XTB simulation input parameters from the initial prompt."""
    messages = state["messages"]
    structured_llm = llm_with_tools.with_structured_output(XTBSimulationInput)
    simulation_params = structured_llm.invoke(messages)
    
    # Return both the original prompt and the structured parameters
    return {"messages": [
        messages[0],  # Keep the original prompt
        {
            "role": "assistant",
            "content": "I've prepared the simulation parameters based on your request.",
            "function_call": simulation_params.dict()  # Store parameters for next node
        }
    ]}

def chatbot(llm_with_tools: ChatOpenAI, state: State):
    """Second node: Uses the simulation parameters to make tool calls."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for function_call in the last message
    if hasattr(last_message, "function_call"):
        simulation_params = last_message.function_call
        return {"messages": [{
            "role": "assistant",
            "content": "Running XTB calculation with the prepared parameters.",
            "tool_calls": [{
                "id": "run_xtb_1",
                "type": "function",
                "function": {
                    "name": "run_xtb_calculation",
                    "arguments": json.dumps(simulation_params)
                }
            }]
        }]}
    
    # For subsequent messages (after tool responses)
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def construct_xtb_graph(tools: list, llm_with_tools: ChatOpenAI):
    # Create a node that will handle running XTB calculations using the provided tools
    tool_node = run_xtb(tools=tools)
    # Initialize a new state graph that will use our State type for managing workflow state
    graph_builder = StateGraph(State)

    # Add a node that sets up initial simulation parameters using the LLM
    graph_builder.add_node("setup_simulation", lambda state: setup_simulation(llm_with_tools, state))
    # Add a node for the chatbot to make decisions about which tools to use
    graph_builder.add_node("chatbot", lambda state: chatbot(llm_with_tools, state))
    # Add the tool node for running XTB calculations
    graph_builder.add_node("tools", tool_node)
    
    # Add conditional routing from chatbot node based on whether tools are needed
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END},
    )

    # Define the workflow:
    graph_builder.add_edge(START, "setup_simulation")  # Start with setup
    graph_builder.add_edge("setup_simulation", "chatbot")  # Let chatbot decide next steps
    graph_builder.add_edge("tools", "chatbot")  # After running tools, let chatbot decide again
    
    # Compile the graph into an executable workflow
    graph = graph_builder.compile()

    return graph