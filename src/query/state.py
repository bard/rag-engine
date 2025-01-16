from typing import Union
from langchain.schema import Document
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    documents: list[Document]
    is_weather_query: bool
    weather_info: str | None
    query: str | None
    location: str | None
