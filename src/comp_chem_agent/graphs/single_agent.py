from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage
import json
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.tools.ASE_tools import (
    molecule_name_to_smiles,
    smiles_to_atomsdata,
    run_ase,
    save_atomsdata_to_file,
    file_to_atomsdata,
    calculate_thermochemistry,
    run_single_point,
)
from comp_chem_agent.prompt.single_agent_prompt import single_agent_prompt
from comp_chem_agent.utils.logging_config import setup_logger

logger = setup_logger(__name__)


class State(TypedDict):
    messages: Annotated[list, add_messages]


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: State) -> State:
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

                print("Output from TOOL CALLING: ", outputs)
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=json.dumps({"error": str(e)}),
                        name=tool_name if tool_name else "unknown_tool",
                        tool_call_id=tool_call.get("id", ""),
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


def ASEAgent(state: State, llm: ChatOpenAI):
    """LLM node that processes messages and decides next actions."""
    tools = [
        file_to_atomsdata,
        smiles_to_atomsdata,
        run_ase,
        molecule_name_to_smiles,
        save_atomsdata_to_file,
        calculate_thermochemistry,
        run_single_point,
    ]
    messages = [
        {"role": "system", "content": single_agent_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def construct_geoopt_graph(llm: ChatOpenAI):
    try:
        logger.info("Constructing geometry optimization graph")
        checkpointer = MemorySaver()
        tools = [
            file_to_atomsdata,
            smiles_to_atomsdata,
            run_ase,
            molecule_name_to_smiles,
            save_atomsdata_to_file,
            calculate_thermochemistry,
            run_single_point,
        ]
        tool_node = BasicToolNode(tools=tools)
        graph_builder = StateGraph(State)
        graph_builder.add_node("ASEAgent", lambda state: ASEAgent(state, llm))
        graph_builder.add_node("tools", tool_node)
        graph_builder.add_conditional_edges(
            "ASEAgent",
            route_tools,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "ASEAgent")
        graph_builder.add_edge(START, "ASEAgent")
        graph = graph_builder.compile(checkpointer=checkpointer)
        logger.info("Graph construction completed")
        return graph
    except Exception as e:
        logger.error(f"Error constructing graph: {str(e)}")
        raise
