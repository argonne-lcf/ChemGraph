from langgraph.graph import END, START, StateGraph
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.models.ASEinput import (
    ASEAtomicInput,
)
from comp_chem_agent.prompt.opt_vib_prompt import *
from comp_chem_agent.tools.ASE_tools import *
from comp_chem_agent.state.opt_vib_state import MultiAgentState
from comp_chem_agent.models.opt_vib_models import (
    RouterResponse,
    QCEngineFeedbackResponse,
    ASEFeedbackResponse,
)
from comp_chem_agent.models.qcengineinput import AtomicInputWrapper
from comp_chem_agent.tools.qcengine_tools import run_qcengine
from comp_chem_agent.tools.ASE_tools import run_ase

import json
import logging

logger = logging.getLogger(__name__)


def first_router_agent(state: MultiAgentState, llm):
    """An LLM router node that decides the workflow based on user's query."""
    prompt = first_router_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    structured_llm = llm.with_structured_output(RouterResponse)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"first_router_response": [response]}


def route_first_router(state: MultiAgentState) -> str:
    """Extract next_agent from first_router_response."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("first_router_response", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to router_tool: {state}")
    try:
        content_dict = json.loads(ai_message.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON content: {ai_message.content}") from e

    next_agent = content_dict["next_agent"]
    return next_agent


def regular_agent(state: MultiAgentState, llm):
    """A LLM agent that handles general queries."""
    prompt = regular_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    response = llm.invoke(messages)
    return {"end_response": [response]}


def ase_geometry_agent(state: MultiAgentState, llm):
    """An LLM node that generates initial molecular structure"""
    prompt = geometry_input_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['geometry_response']}"},
    ]
    tools = [file_to_atomsdata, molecule_name_to_smiles, smiles_to_atomsdata]
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)
    return {"geometry_response": [response]}


def qcengine_geometry_agent(state: MultiAgentState, llm):
    """An LLM node that generates initial molecular structure"""
    prompt = geometry_input_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['geometry_response']}"},
    ]
    tools = [file_to_atomsdata, molecule_name_to_smiles, smiles_to_atomsdata]
    llm_with_tools = llm.bind_tools(tools=tools)
    response = llm_with_tools.invoke(messages)
    return {"geometry_response": [response]}


def route_tools_geometry(state: MultiAgentState) -> str:
    """Route tools for geometry agents"""
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


class ASEGeometryTool:
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
                    else (
                        tool_result
                        if isinstance(tool_result, dict)
                        else str(tool_result)
                    )
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


class QCEngineGeometryTool:
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
                    else (
                        tool_result
                        if isinstance(tool_result, dict)
                        else str(tool_result)
                    )
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


def ase_parameter_agent(state: MultiAgentState, llm: ChatOpenAI):
    if len(state["feedback_response"]) == 0:
        feedback = []
    else:
        feedback = state["feedback_response"][-1].content
    parameter_prompt = ase_parameters_input_prompt.format(
        geometry_response=state["geometry_response"][-1].content, feedback=feedback
    )
    messages = [
        {"role": "system", "content": parameter_prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    structured_llm = llm.with_structured_output(ASEAtomicInput)
    response = structured_llm.invoke(messages).model_dump_json()
    logger.debug(f"ASE Parameters Prompt: {parameter_prompt}")
    return {"parameter_response": response}


def qcengine_parameter_agent(state: MultiAgentState, llm: ChatOpenAI):
    if len(state["feedback_response"]) == 0:
        feedback = []
    else:
        feedback = state["feedback_response"][-1].content
    prompt = qcengine_parameter_prompt.format(
        geometry_response=state["geometry_response"][-1], feedback=feedback
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    structured_llm = llm.with_structured_output(AtomicInputWrapper)
    response = structured_llm.invoke(messages).model_dump_json()

    print("RESPONSE", response)
    return {"parameter_response": response}


def ase_feedback_agent(state: MultiAgentState, llm):
    prompt = ase_feedback_prompt.format(aseoutput=state["opt_response"][-1].content)
    messages = [
        {"role": "system", "content": prompt},
    ]
    structured_llm = llm.with_structured_output(ASEFeedbackResponse)
    response = structured_llm.invoke(messages)
    return {"feedback_response": [response.model_dump_json()]}


def qcengine_feedback_agent(state: MultiAgentState, llm):
    prompt = qcengine_feedback_prompt.format(
        qcengine_output=state["opt_response"][-1].content
    )
    messages = [
        {"role": "system", "content": prompt},
    ]
    structured_llm = llm.with_structured_output(QCEngineFeedbackResponse)
    response = structured_llm.invoke(messages)
    return {"feedback_response": [response.model_dump_json()]}


def route_feedback(state: MultiAgentState) -> str:
    """Extract next_agent from first_router_response."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("feedback_response", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to router_tool: {state}")
    try:
        content_dict = json.loads(ai_message.content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON content: {ai_message.content}") from e

    next_agent = content_dict["next_agent"]
    return next_agent


def end_agent(state: MultiAgentState, llm):
    prompt = end_prompt.format(
        output=state["opt_response"][-1].content,
        feedback=state["feedback_response"][-1].content,
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    response = llm.invoke(messages)
    return {"end_response": [response]}


def construct_opt_vib_graph(llm: ChatOpenAI):
    checkpointer = MemorySaver()
    # Nodes
    graph_builder = StateGraph(MultiAgentState)
    graph_builder.add_node("RouterAgent", lambda state: first_router_agent(state, llm))
    graph_builder.add_node("RegularAgent", lambda state: regular_agent(state, llm))

    graph_builder.add_node(
        "ASEGeometryAgent", lambda state: ase_geometry_agent(state, llm)
    )
    graph_builder.add_node(
        "QCEngineGeometryAgent", lambda state: qcengine_geometry_agent(state, llm)
    )

    graph_builder.add_node(
        "ASEGeometryTool",
        ASEGeometryTool(
            tools=[molecule_name_to_smiles, smiles_to_atomsdata, file_to_atomsdata]
        ),
    )
    graph_builder.add_node(
        "QCEngineGeometryTool",
        QCEngineGeometryTool(
            tools=[molecule_name_to_smiles, smiles_to_atomsdata, file_to_atomsdata]
        ),
    )

    graph_builder.add_node(
        "ASEParameterAgent", lambda state: ase_parameter_agent(state, llm)
    )
    graph_builder.add_node(
        "QCEngineParameterAgent", lambda state: qcengine_parameter_agent(state, llm)
    )

    graph_builder.add_node("RunQCEngine", lambda state: run_qcengine(state))
    graph_builder.add_node("RunASE", lambda state: run_ase(state))

    graph_builder.add_node(
        "ASEFeedbackAgent", lambda state: ase_feedback_agent(state, llm)
    )
    graph_builder.add_node(
        "QCEngineFeedbackAgent", lambda state: qcengine_feedback_agent(state, llm)
    )

    graph_builder.add_node("EndAgent", lambda state: end_agent(state, llm))

    # Edges
    graph_builder.add_edge(START, "RouterAgent")
    graph_builder.add_conditional_edges(
        "RouterAgent",
        route_first_router,
        {
            "ASEWorkflow": "ASEGeometryAgent",
            "RegularAgent": "RegularAgent",
            "QCEngineWorkflow": "QCEngineGeometryAgent",
        },
    )
    graph_builder.add_edge("RegularAgent", END)
    graph_builder.add_conditional_edges(
        "ASEGeometryAgent",
        route_tools_geometry,
        {"tools": "ASEGeometryTool", "next": "ASEParameterAgent"},
    )
    graph_builder.add_conditional_edges(
        "QCEngineGeometryAgent",
        route_tools_geometry,
        {"tools": "QCEngineGeometryTool", "next": "QCEngineParameterAgent"},
    )

    graph_builder.add_edge("QCEngineGeometryTool", "QCEngineGeometryAgent")
    graph_builder.add_edge("ASEGeometryTool", "ASEGeometryAgent")

    graph_builder.add_edge("ASEParameterAgent", "RunASE")
    graph_builder.add_edge("QCEngineParameterAgent", "RunQCEngine")

    graph_builder.add_edge("RunASE", "ASEFeedbackAgent")
    graph_builder.add_edge("RunQCEngine", "QCEngineFeedbackAgent")

    graph_builder.add_conditional_edges(
        "ASEFeedbackAgent",
        route_feedback,
        {"ASEParameterAgent": "ASEParameterAgent", "EndAgent": "EndAgent"},
    )
    graph_builder.add_conditional_edges(
        "QCEngineFeedbackAgent",
        route_feedback,
        {"QCEngineParameterAgent": "QCEngineParameterAgent", "EndAgent": "EndAgent"},
    )
    graph_builder.add_edge("EndAgent", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph
