import pprint
from pydantic import BaseModel
from fastapi import APIRouter, Query, Depends
from langchain.schema import HumanMessage

from ..config import Config
from .. import workflow_query
from .deps import get_config


router = APIRouter()


class QueryResponse(BaseModel):
    """Response model for query endpoint"""

    answer: str
    sources: list[str]


@router.get("/query", response_model=QueryResponse, operation_id="query")
def query(
    q: str = Query(description="The question to ask the knowledge base"),
    topic_id: str | None = Query(description="Optional topic id"),
    config: Config = Depends(get_config),
):
    """Query the knowledge base with a question"""
    graph = workflow_query.get_graph()

    # Initialize state with the user's question
    # TODO: we might want the entire chat history instead here
    initial_state = workflow_query.AgentState(
        messages=[HumanMessage(content=q)],
        retrieved_knowledge=[],
        query=None,
        topic_id=topic_id,
        external_knowledge_sources=[],
    )

    result = graph.invoke(initial_state, config.to_runnable_config())

    return {
        "answer": result["messages"][-1].content,
        "sources": [doc.id for doc in result["retrieved_knowledge"]],
    }
