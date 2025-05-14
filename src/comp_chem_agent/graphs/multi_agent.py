from langgraph.graph import END, START, StateGraph
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.prompt.multi_agent_prompt import (
    first_router_prompt,
    regular_prompt,
    geometry_input_prompt,
    new_ase_parameters_input_prompt,
    qcengine_parameter_prompt,
    ase_feedback_prompt,
    qcengine_feedback_prompt,
    end_prompt,
)
from comp_chem_agent.tools.ASE_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
    file_to_atomsdata,
    run_ase_with_state,
)
from comp_chem_agent.state.state import MultiAgentState
from comp_chem_agent.models.multi_agent_response import (
    RouterResponse,
    QCEngineFeedbackResponse,
    ASEFeedbackResponse,
)
from comp_chem_agent.models.qcengine_input import QCEngineInputSchema
from comp_chem_agent.tools.qcengine_tools import run_qcengine_multi_framework
from comp_chem_agent.models.ase_input import ASEInputSchema
import json
import logging

logger = logging.getLogger(__name__)


def first_router_agent(state: MultiAgentState, llm):
    """An LLM router node that decides the workflow based on user's query.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing the user's question
    llm : ChatOpenAI
        The language model to use for routing

    Returns
    -------
    dict
        Updated state containing the router's response
    """
    prompt = first_router_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    structured_llm = llm.with_structured_output(RouterResponse)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"first_router_response": [response]}


def route_first_router(state: MultiAgentState) -> str:
    """Extract next_agent from first_router_response.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing the router's response

    Returns
    -------
    str
        The name of the next agent to route to

    Raises
    ------
    ValueError
        If no messages are found in the input state
        If there is an error decoding the JSON content
    """
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
    """A LLM agent that handles general queries.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing the user's question
    llm : ChatOpenAI
        The language model to use for processing

    Returns
    -------
    dict
        Updated state containing the agent's response
    """
    prompt = regular_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    response = llm.invoke(messages)
    return {"end_response": [response]}


def geometry_agent(state: MultiAgentState, llm):
    """An LLM node that generates initial molecular structure.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing geometry-related information
    llm : ChatOpenAI
        The language model to use for structure generation

    Returns
    -------
    dict
        Updated state containing the generated geometry response
    """
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
    """Route tools for geometry agents.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing geometry-related information

    Returns
    -------
    str
        Either 'tools' or the next agent based on the state

    Raises
    ------
    ValueError
        If no messages are found in the input state
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("geometry_response", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    has_tool_calls = (
        hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0
    )
    if has_tool_calls:
        return "tools"
    else:
        return route_first_router(state)


class GeometryTool:
    """A node that executes tools requested in the last AIMessage for geometry operations.

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

    def __call__(self, inputs: MultiAgentState) -> MultiAgentState:
        """Execute tools requested in the last message.

        Parameters
        ----------
        inputs : MultiAgentState
            The current state containing messages

        Returns
        -------
        MultiAgentState
            Updated state containing tool execution results

        Raises
        ------
        ValueError
            If no message is found in the input state
        """
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
    """An LLM agent that generates ASE parameters based on geometry and feedback.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing geometry and feedback information
    llm : ChatOpenAI
        The language model to use for parameter generation

    Returns
    -------
    dict
        Updated state containing the generated parameters
    """
    if len(state["feedback_response"]) == 0:
        feedback = []
    else:
        feedback = state["feedback_response"][-1].content
    prompt = new_ase_parameters_input_prompt.format(
        geometry_response=state["geometry_response"][-1].content,
        feedback=feedback,
        ase_schema=ASEInputSchema.model_json_schema(),
    )
    logger.debug(f"PROMPT: {prompt}")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    structured_llm = llm.with_structured_output(ASEInputSchema)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"parameter_response": response}


def qcengine_parameter_agent(state: MultiAgentState, llm: ChatOpenAI):
    """An LLM agent that generates QCEngine parameters based on geometry and feedback.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing geometry and feedback information
    llm : ChatOpenAI
        The language model to use for parameter generation

    Returns
    -------
    dict
        Updated state containing the generated parameters
    """
    if len(state["feedback_response"]) == 0:
        feedback = []
    else:
        feedback = state["feedback_response"][-1].content
    prompt = qcengine_parameter_prompt.format(
        geometry_response=state["geometry_response"][-1].content,
        feedback=feedback,
        qcengine_schema=QCEngineInputSchema.model_json_schema(),
    )

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"},
    ]
    structured_llm = llm.with_structured_output(QCEngineInputSchema)
    response = structured_llm.invoke(messages).model_dump_json()
    print(response)
    return {"parameter_response": response}


def ase_feedback_agent(state: MultiAgentState, llm):
    """An LLM agent that generates feedback based on ASE optimization results.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing ASE optimization results
    llm : ChatOpenAI
        The language model to use for feedback generation

    Returns
    -------
    dict
        Updated state containing the feedback response
    """
    prompt = ase_feedback_prompt.format(aseoutput=state["opt_response"][-1].content)
    messages = [
        {"role": "system", "content": prompt},
    ]
    structured_llm = llm.with_structured_output(ASEFeedbackResponse)
    response = structured_llm.invoke(messages)
    return {"feedback_response": [response.model_dump_json()]}


def qcengine_feedback_agent(state: MultiAgentState, llm):
    """An LLM agent that generates feedback based on QCEngine results.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing QCEngine results
    llm : ChatOpenAI
        The language model to use for feedback generation

    Returns
    -------
    dict
        Updated state containing the feedback response
    """
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
    """Extract next_agent from feedback response.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing feedback information

    Returns
    -------
    str
        The name of the next agent to route to

    Raises
    ------
    ValueError
        If no messages are found in the input state
        If there is an error decoding the JSON content
    """
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
    """An LLM agent that generates the final response.

    Parameters
    ----------
    state : MultiAgentState
        The current state containing optimization results and feedback
    llm : ChatOpenAI
        The language model to use for final response generation

    Returns
    -------
    dict
        Updated state containing the final response
    """
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


def construct_multi_framework_graph(llm: ChatOpenAI):
    """Construct a graph for multi-framework computational chemistry workflow.

    This function creates a state graph that implements a workflow for computational
    chemistry tasks using multiple frameworks (ASE and QCEngine).

    Parameters
    ----------
    llm : ChatOpenAI
        The language model to use in the workflow

    Returns
    -------
    StateGraph
        A compiled state graph implementing the multi-framework workflow
    """
    checkpointer = MemorySaver()
    # Nodes
    graph_builder = StateGraph(MultiAgentState)
    graph_builder.add_node("RouterAgent", lambda state: first_router_agent(state, llm))
    graph_builder.add_node("RegularAgent", lambda state: regular_agent(state, llm))

    graph_builder.add_node("GeometryAgent", lambda state: geometry_agent(state, llm))
    graph_builder.add_node(
        "GeometryTool",
        GeometryTool(
            tools=[molecule_name_to_smiles, smiles_to_atomsdata, file_to_atomsdata]
        ),
    )

    graph_builder.add_node(
        "ASEParameterAgent", lambda state: ase_parameter_agent(state, llm)
    )
    graph_builder.add_node(
        "QCEngineParameterAgent", lambda state: qcengine_parameter_agent(state, llm)
    )

    graph_builder.add_node(
        "RunQCEngine", lambda state: run_qcengine_multi_framework(state)
    )
    graph_builder.add_node("RunASE", lambda state: run_ase_with_state(state))

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
            "ASEWorkflow": "GeometryAgent",
            "RegularAgent": "RegularAgent",
            "QCEngineWorkflow": "GeometryAgent",
        },
    )
    graph_builder.add_edge("RegularAgent", END)
    graph_builder.add_conditional_edges(
        "GeometryAgent",
        route_tools_geometry,
        {
            "tools": "GeometryTool",
            "ASEWorkflow": "ASEParameterAgent",
            "QCEngineWorkflow": "QCEngineParameterAgent",
        },
    )
    graph_builder.add_edge("GeometryTool", "GeometryAgent")
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
