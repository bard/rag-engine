from fastapi import APIRouter, Query, Depends
from langchain.schema import HumanMessage

from ..config import Config
from .. import workflow_query
from .deps import get_config


router = APIRouter()


@router.get("/query")
def query(
    q: str = Query(description="The question to ask the knowledge base"),
    config: Config = Depends(get_config),
):
    """Query the knowledge base with a question"""
    graph = workflow_query.get_graph()

    # Initialize state with the user's question
    # TODO: we might want the entire chat history instead here
    initial_state = {
        "messages": [HumanMessage(content=q)],
        "documents": [],
        "query": None,
        "location": None,
        "external_knowledge_sources": [],
    }

    result = graph.invoke(initial_state, config.to_runnable_config())

    # Return the last message from the conversation
    return {"answer": result["messages"][-1].content}
