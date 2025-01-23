from typing import TypedDict
from langchain.schema import Document
from langchain_core.runnables import RunnableConfig

from ..config import Config
from ..services import get_vector_store
from .state import AgentState


class RetrieveStateUpdate(TypedDict):
    documents: list[Document]


def retrieve(state: AgentState, config: RunnableConfig) -> RetrieveStateUpdate:
    c = Config.from_runnable_config(config)
    vector_store = get_vector_store(c)
    query = state["query"]
    assert query is not None

    documents_with_scores = vector_store.similarity_search_with_relevance_scores(
        query, k=2
    )

    filtered_documents = [
        doc
        for doc, score in documents_with_scores
        if score >= c.vector_store.score_threshold
    ]

    return {"documents": filtered_documents}
