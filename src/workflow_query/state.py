from typing import TypedDict, Literal
from langchain.schema import Document
from langgraph.graph import MessagesState


class WeatherExternalKnowledgeSource(TypedDict):
    type: Literal["weather"]
    location: str
    # TODO add 'time' to support forecasts


# TODO stub only, not used
class NewsExternalKnowledgeSource(TypedDict):
    type: Literal["news"]
    location: str


ExternalKnowledgeSource = WeatherExternalKnowledgeSource | NewsExternalKnowledgeSource


class AgentState(MessagesState):
    documents: list[Document]
    query: str | None
    location: str | None
    external_knowledge_sources: list[ExternalKnowledgeSource]
