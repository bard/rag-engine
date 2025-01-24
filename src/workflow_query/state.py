from typing import TypedDict, Literal
from langchain.schema import Document
from langgraph.graph import MessagesState


class WeatherExternalKnowledgeSource(TypedDict):
    type: Literal["weather"]
    location: str
    # TODO add 'time' to support forecasts


# TODO demo purpose only, not used
class NewsExternalKnowledgeSource(TypedDict):
    type: Literal["news"]


ExternalKnowledgeSource = WeatherExternalKnowledgeSource | NewsExternalKnowledgeSource


class AgentState(MessagesState):
    query: str | None
    topic_id: str | None
    retrieved_knowledge: list[Document]
    external_knowledge_sources: list[ExternalKnowledgeSource]
