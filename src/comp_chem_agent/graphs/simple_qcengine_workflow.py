from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from comp_chem_agent.models.qcengineinput import AtomicInputWrapper


class State(TypedDict):
    messages: Annotated[list, add_messages]


def chatbot(state: State, llm: ChatOpenAI):
    """LLM node that processes messages and decides next actions."""
    messages = state["messages"]
    structure_llm = llm.with_structured_output(AtomicInputWrapper)
    # structure_llm = llm.with_structured_output(AtomicInput)
    response = structure_llm.invoke(messages).model_dump_json()
    return {"messages": [response]}


def construct_qcengine_graph(llm: ChatOpenAI):
    checkpointer = MemorySaver()
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", lambda state: chatbot(state, llm))
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph
