import pprint
from typing import TypedDict
from langchain.schema import Document
from langchain_core.runnables import RunnableConfig

from .. import services
from ..config import Config
from .state import AgentState


class RetrieveStateUpdate(TypedDict):
    retrieved_knowledge: list[Document]


def retrieve_from_knowledge_base(
    state: AgentState, config: RunnableConfig
) -> RetrieveStateUpdate:
    conf = Config.from_runnable_config(config)

    query = state["query"]
    assert query is not None

    if state["topic_id"] is None:
        metadata_filter = {"topic_id": "UNCATEGORIZED"}
    else:
        metadata_filter = {"topic_id": state["topic_id"]}

    documents_with_scores = services.get_vector_store(
        conf
    ).similarity_search_with_relevance_scores(query, k=2, filter=metadata_filter)

    most_relevant_documents = [
        doc
        for doc, score in documents_with_scores
        if score >= conf.vector_store.score_threshold
    ]

    return {
        "retrieved_knowledge": [*state["retrieved_knowledge"], *most_relevant_documents]
    }
