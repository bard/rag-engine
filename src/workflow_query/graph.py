import pprint
from typing import Literal
from langgraph.graph import StateGraph, END

from ..config import Config
from .state import AgentState
from .node_rerank import rerank
from .node_retrieve import retrieve
from .node_fetch_weather_info import fetch_weather_info
from .node_classify_query import classify_query
from .node_generate import generate


def route_based_on_classification(
    state: AgentState,
) -> Literal["fetch_weather_info", "retrieve"]:
    if state.get("is_weather_query"):
        return "fetch_weather_info"
    return "retrieve"


def get_graph():
    builder = StateGraph(AgentState, Config)
    builder.set_entry_point("classify_query")
    builder.add_node("classify_query", classify_query)
    builder.add_node("fetch_weather_info", fetch_weather_info)
    builder.add_node("retrieve", retrieve)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate)
    builder.add_conditional_edges(
        "classify_query",
        route_based_on_classification,
        {
            "fetch_weather_info": "fetch_weather_info",
            "retrieve": "retrieve",
        },
    )
    builder.add_edge("fetch_weather_info", "retrieve")
    builder.add_edge("retrieve", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)

    graph = builder.compile()
    return graph
