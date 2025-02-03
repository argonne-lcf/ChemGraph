from comp_chem_agent.models.atomsdata import AtomsData
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import END, START, add_messages, StateGraph
from langchain_core.messages import ToolMessage
import json
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.models.ASEinput import ASESimulationInput, ASESimulationOutput
from comp_chem_agent.prompt.prompt import geometry_input_prompt, parameters_input_prompt, execution_prompt, feedback_prompt, router_prompt, end_prompt, planner_prompt
from comp_chem_agent.tools.ASE_tools import *
from comp_chem_agent.models.agent_response import RouterResponse, PlannerResponse
class MultiAgentState(TypedDict):
    question: str
    planner_response: Annotated[list, add_messages]
    regular_response: Annotated[list, add_messages]
    geometry_response: Annotated[list, add_messages]
    parameter_response: Annotated[list, add_messages]
    opt_response: Annotated[list, add_messages]
    feedback_response: Annotated[list, add_messages]
    router_response: Annotated[list, add_messages]
    end_response: Annotated[list, add_messages]

def PlannerAgent(state: MultiAgentState, llm):
    prompt = planner_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"}]
    structured_llm = llm.with_structured_output(PlannerResponse)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"planner_response": [response]}

def RegularAgent(state: MultiAgentState, llm):
    prompt = """You are a helpful assistant"""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"}]
    response = llm.invoke(messages)
    return {"regular_response": [response]}

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
                    tool_result.model_dump() if hasattr(tool_result, "dict")
                    else tool_result if isinstance(tool_result, dict)
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
            return {
                "geometry_response": outputs
            }

class GeometryOptimizationTool:
    """A node that runs the tools requested in the last AIMessage."""
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: MultiAgentState) -> MultiAgentState:
        if messages := inputs.get("opt_response", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        
        outputs = []
        print("TOOL CALLS", message.tool_calls)
        for tool_call in message.tool_calls:
            try:
                tool_name = tool_call.get("name")
                if not tool_name or tool_name not in self.tools_by_name:
                    raise ValueError(f"Invalid tool name: {tool_name}")
                tool_args = tool_call.get("args", {})
                tool_result = self.tools_by_name[tool_name].invoke(tool_args)
                result_content = (
                    tool_result.model_dump() if hasattr(tool_result, "dict")
                    else tool_result if isinstance(tool_result, dict)
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
            return {
                "opt_response": outputs,
            }

def GeometryInputAgent(state: MultiAgentState, llm):
    messages = [
            {"role": "system", "content": geometry_input_prompt},
            {"role": "user", "content": f"{state['geometry_response']}"}]
    tools = [file_to_atomsdata, molecule_name_to_smiles, smiles_to_atomsdata]
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)
    return {"geometry_response": [response]}

def ParameterInputAgent(state: MultiAgentState, llm: ChatOpenAI ):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if len(state['feedback_response']) == 0:
        feedback = []
    else:
        feedback = state['feedback_response'][-1]
    parameter_prompt = parameters_input_prompt.format(geometry_response=state['geometry_response'], feedback=feedback)
    print("PARAMETER PROMPT: ", parameter_prompt)
    messages = [
            {"role": "system", "content": parameter_prompt},
            {"role": "user", "content": f"{state['parameter_response']}"}
    ]
    structured_llm  = llm.with_structured_output(ASESimulationInput)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"parameter_response": response}

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
        hasattr(ai_message, "tool_calls") and 
        len(ai_message.tool_calls) > 0
    )
    return "tools" if has_tool_calls else "next"

def route_tools_opt(state: MultiAgentState) -> str:
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
    elif messages := state.get("opt_response", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    # Check if there are tool calls to process
    has_tool_calls = (
        hasattr(ai_message, "tool_calls") and 
        len(ai_message.tool_calls) > 0
    )
    return "tools" if has_tool_calls else "next"

def GeometryOptimizationAgent(state: MultiAgentState, llm): 
    llm_with_tools = llm.bind_tools(tools=[geometry_optimization])
    opt_prompt = execution_prompt.format(parameters=state['parameter_response'][-1])
    messages = [
            {"role": "system", "content": opt_prompt},
            {"role": "user", "content": f"{state['question']}"}]
    response = llm_with_tools.invoke(messages)
    return {"opt_response": [response]}

def FeedbackAgent(state: MultiAgentState, llm):
    prompt = feedback_prompt.format(aseoutput=state['opt_response'][-1])
    messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{state['question']}"}]
    response = llm.invoke(messages)
    return {"feedback_response": [response]}

def RouterAgent(state: MultiAgentState, llm):
    prompt = router_prompt.format(feedback=state['feedback_response'][-1])
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"}]
    structured_llm = llm.with_structured_output(RouterResponse)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"router_response": [response]}

def router_tool(state: MultiAgentState) -> str:
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
    next_agent = content_dict['next_agent']
    return next_agent

def router_planner(state: MultiAgentState) -> str:
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("planner_response", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to router_tool: {state}")
    # Parse the content into a dictionary
    try:
        content_dict = json.loads(ai_message.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON content: {ai_message.content}") from e    
    # Access the 'next_agent' field
    next_agent = content_dict['next_agent']
    return next_agent

def EndAgent(state: MultiAgentState, llm):
    prompt = end_prompt.format(state=state)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"}]
    response = llm.invoke(messages)
    return {"end_response": [response]}

def construct_ase_graph(llm: ChatOpenAI):
    checkpointer = MemorySaver()
    # Nodes
    graph_builder = StateGraph(MultiAgentState)
    graph_builder.add_node("PlannerAgent", lambda state: PlannerAgent(state, llm))
    graph_builder.add_node("RegularAgent", lambda state: RegularAgent(state, llm))
    graph_builder.add_node("GeometryInputAgent", lambda state: GeometryInputAgent(state, llm))
    graph_builder.add_node("GeometryInputTool", GeometryInputTool(tools=[molecule_name_to_smiles, smiles_to_atomsdata, file_to_atomsdata]))
    graph_builder.add_node("ParameterInputAgent", lambda state: ParameterInputAgent(state, llm))
    graph_builder.add_node("GeometryOptimizationAgent", lambda state: GeometryOptimizationAgent(state, llm))
    graph_builder.add_node("GeometryOptimizationTool", GeometryOptimizationTool(tools=[geometry_optimization]))
    graph_builder.add_node("FeedbackAgent", lambda state: FeedbackAgent(state, llm))
    graph_builder.add_node("RouterAgent", lambda state: RouterAgent(state, llm))
    graph_builder.add_node("EndAgent", lambda state: EndAgent(state, llm))
    # Edges
    #graph_builder.add_edge(START, "GeometryInputAgent")
    graph_builder.add_edge(START, "PlannerAgent")
    graph_builder.add_conditional_edges(
        "PlannerAgent", router_planner, 
        {"WorkflowAgent": "GeometryInputAgent", "RegularAgent": "RegularAgent"})
    graph_builder.add_conditional_edges(
        "GeometryInputAgent", route_tools_geometry,
        {"tools": "GeometryInputTool", "next": "ParameterInputAgent"})
    graph_builder.add_edge("GeometryInputTool", "GeometryInputAgent")
    graph_builder.add_edge("ParameterInputAgent", "GeometryOptimizationAgent")
    graph_builder.add_edge("GeometryOptimizationAgent", "GeometryOptimizationTool")
    graph_builder.add_edge("GeometryOptimizationTool", "FeedbackAgent")
    graph_builder.add_edge("FeedbackAgent", "RouterAgent")
    graph_builder.add_conditional_edges("RouterAgent", router_tool)
    graph_builder.add_edge("EndAgent", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph
