from langgraph.graph import END, START, StateGraph
from comp_chem_agent.state.state import MultiAgentState

from langchain_core.messages import ToolMessage
import json
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.prompt.prompt import (
    geometry_input_prompt,
    planner_prompt,
)
from comp_chem_agent.tools.ASE_tools import *
from comp_chem_agent.models.agent_response import PlannerResponse
from comp_chem_agent.tools.qcengine_tools import run_qcengine
from comp_chem_agent.models.qcengineinput import AtomicInputWrapper

router_prompt = """
You are a routing agent responsible for directing the conversation to the appropriate next agent based on the user's question. 

### Available Agents:
1. **WorkflowAgent**: Executes structured workflows for geometry optimization using the QCEngine.
2. **RegularAgent**: Handles general inquiries that do not require workflow execution.

### Routing Criteria:
- Assign the query to **WorkflowAgent** if it involves performing a workflow related to QCEngine.
- Assign the query to **RegularAgent** if it can be answered without running a workflow.

Ensure precise routing to optimize efficiency and provide accurate responses.
"""

parameters_input_prompt = """
You are an expert in computational chemistry and proficient in using QCEngine library. Your task is to configure simulation parameters based on the user's request and feedback.

1. Source of Structure Data:

If feedback is provided, use it to adjust the simulation parameters and update the atomsdata.
If no feedback is available, retrieve the atomsdata from the geometry agent's response.
Geometry agent response: {geometry_response}
"""


def RouterAgent(state: MultiAgentState, llm):
    prompt = planner_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    structured_llm = llm.with_structured_output(PlannerResponse)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"router_response": [response]}


def router_router(state: MultiAgentState) -> str:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("router_response", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to router_tool: {state}")
    # Parse the content into a dictionary
    try:
        content_dict = json.loads(ai_message.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON content: {ai_message.content}") from e
    # Access the 'next_agent' field
    next_agent = content_dict["next_agent"]
    return next_agent


def RegularAgent(state: MultiAgentState, llm):
    prompt = """You are a helpful assistant"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    response = llm.invoke(messages)
    return {"regular_response": [response]}


def GeometryInputAgent(state: MultiAgentState, llm):
    prompt = geometry_input_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['geometry_response']}"},
    ]
    tools = [file_to_atomsdata, molecule_name_to_smiles, smiles_to_atomsdata]
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)
    return {"geometry_response": [response]}


class GeometryInputTool:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: MultiAgentState) -> MultiAgentState:
        if messages := inputs.get("geometry_response", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")

        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name")
                if not tool_name or tool_name not in self.tools_by_name:
                    raise ValueError(f"Invalid tool name: {tool_name}")
                tool_args = tool_call.get("args", {})
                tool_result = self.tools_by_name[tool_name].invoke(tool_args)
                result_content = (
                    tool_result.model_dump()
                    if hasattr(tool_result, "dict")
                    else tool_result
                    if isinstance(tool_result, dict)
                    else str(tool_result)
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
            return {"geometry_response": outputs}


def route_tools_geometry(state: MultiAgentState) -> str:
    """
    Routes the workflow based on whether tool calls are present in the last message.
    Args:
        state (MultiAgentState): Current state of the workflow
    Returns:
        str: Either "tools" if tool calls are present, or END if no tools needed
    Raises:
        ValueError: If no messages are found in the state
    """
    # Get the last AI message from either list or dict state
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("geometry_response", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    # Check if there are tool calls to process
    has_tool_calls = (
        hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0
    )
    return "tools" if has_tool_calls else "next"


def ParameterInputAgent(state: MultiAgentState, llm: ChatOpenAI):
    parameter_prompt = parameters_input_prompt.format(
        geometry_response=state["geometry_response"]
    )
    messages = [
        {"role": "system", "content": parameter_prompt},
        {"role": "user", "content": f"{state['parameter_response']}"},
    ]
    structured_llm = llm.with_structured_output(AtomicInputWrapper)
    response = structured_llm.invoke(messages).model_dump_json()
    print("Parameter RESPONSE:", response)
    return {"parameter_response": response}


def construct_qcengine_graph(llm: ChatOpenAI):
    checkpointer = MemorySaver()
    # Nodes
    graph_builder = StateGraph(MultiAgentState)
    graph_builder.add_node("RouterAgent", lambda state: RouterAgent(state, llm))
    graph_builder.add_node("RegularAgent", lambda state: RegularAgent(state, llm))
    graph_builder.add_node(
        "GeometryInputAgent", lambda state: GeometryInputAgent(state, llm)
    )
    graph_builder.add_node(
        "GeometryInputTool",
        GeometryInputTool(
            tools=[molecule_name_to_smiles, smiles_to_atomsdata, file_to_atomsdata]
        ),
    )
    graph_builder.add_node(
        "ParameterInputAgent", lambda state: ParameterInputAgent(state, llm)
    )
    graph_builder.add_node("OptimizationTool", lambda state: run_qcengine(state))

    # Edges
    graph_builder.add_edge(START, "RouterAgent")
    graph_builder.add_conditional_edges(
        "RouterAgent",
        router_router,
        {"WorkflowAgent": "GeometryInputAgent", "RegularAgent": "RegularAgent"},
    )
    graph_builder.add_conditional_edges(
        "GeometryInputAgent",
        route_tools_geometry,
        {"tools": "GeometryInputTool", "next": "ParameterInputAgent"},
    )
    graph_builder.add_edge("GeometryInputTool", "GeometryInputAgent")
    graph_builder.add_edge("ParameterInputAgent", "OptimizationTool")
    graph_builder.add_edge("OptimizationTool", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph
