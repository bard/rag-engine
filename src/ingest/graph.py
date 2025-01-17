from langgraph.graph import StateGraph, START, END
from ..config import AgentConfig
from .state import AgentState
from .node_extract import extract
from .node_fetch import fetch
from .node_index_and_store import index_and_store


def get_graph():
    builder = StateGraph(AgentState, AgentConfig)
    builder.add_node("fetch", fetch)  # pyright: ignore
    builder.add_node("extract", extract)  # pyright: ignore
    builder.add_node("ingest", index_and_store)  # pyright: ignore
    builder.add_edge(START, "fetch")
    builder.add_edge("fetch", "extract")
    builder.add_edge("extract", "ingest")
    builder.add_edge("ingest", END)
    graph = builder.compile()
    return graph
