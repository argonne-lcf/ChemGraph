from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
import json

from comp_chem_agent.prompt.gcmc_prompt import *
from comp_chem_agent.models.gcmc_models import PlannerResponse
class MultiAgentState(TypedDict):
    question: str
    planner_response: Annotated[list, add_messages]
    
def PlannerAgent(state: MultiAgentState, llm):
    prompt = planner_prompt
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"{state['question']}"}]
    structured_llm = llm.with_structured_output(PlannerResponse)
    response = structured_llm.invoke(messages).model_dump_json()
    return {"planner_response": [response]}

def DataQueryAgent(state: MultiAgentState, llm):
    prompt = data_query_prompt

def construct_gcmc_graph(llm: ChatOpenAI):
    checkpointer = MemorySaver()
    # Nodes
    graph_builder = StateGraph(MultiAgentState)
    graph_builder.add_node("PlannerAgent", lambda state: PlannerAgent(state, llm))
    # Edges
    graph_builder.add_edge(START, "PlannerAgent")
    graph_builder.add_edge("PlannerAgent", END)
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph

