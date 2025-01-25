import pprint
from typing import Literal
from langgraph.graph import StateGraph, END

from ..config import Config
from .state import AgentState
from .node_retrieve_from_knowledge_base import retrieve_from_knowledge_base
from .node_retrieve_from_weather_service import retrieve_from_weather_service
from .node_classify_query import classify_query
from .node_generate import generate
from .node_rerank__STUB import rerank


def route_based_on_query_classification(
    state: AgentState,
) -> Literal["retrieve_from_weather_service", "retrieve_from_knowledge_base"]:
    if any(
        source["type"] == "weather" for source in state["external_knowledge_sources"]
    ):
        return "retrieve_from_weather_service"
    else:
        return "retrieve_from_knowledge_base"


def get_graph():
    builder = StateGraph(AgentState, Config)
    builder.set_entry_point("classify_query")
    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_from_weather_service", retrieve_from_weather_service)
    builder.add_node("retrieve_from_knowledge_base", retrieve_from_knowledge_base)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate)
    builder.add_conditional_edges(
        "classify_query",
        route_based_on_query_classification,
        {
            "retrieve_from_weather_service": "retrieve_from_weather_service",
            "retrieve_from_knowledge_base": "retrieve_from_knowledge_base",
        },
    )
    builder.add_edge("retrieve_from_weather_service", "retrieve_from_knowledge_base")
    builder.add_edge("retrieve_from_knowledge_base", "rerank")
    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", END)
    graph = builder.compile()
    return graph
