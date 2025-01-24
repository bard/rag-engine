from typing import TypedDict
from langchain_core.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from .. import services
from ..db import SqlTopic
from ..config import Config
from ..workflow_query.state import AgentState, ExternalKnowledgeSource


class ClassifyStateUpdate(TypedDict):
    query: str
    external_knowledge_sources: list[ExternalKnowledgeSource]


def classify_query(state: AgentState, config: RunnableConfig) -> ClassifyStateUpdate:
    """Detects if the topic is a location and the query is weather-related."""

    conf = Config.from_runnable_config(config)
    db = services.get_db(conf)

    last_message = state["messages"][-1]
    assert isinstance(last_message, HumanMessage)
    query = last_message.content
    assert isinstance(query, str)

    if state["topic_id"] is None:
        return {"query": query, "external_knowledge_sources": []}

    with Session(db) as session:
        topic = session.query(SqlTopic).filter(SqlTopic.id == state["topic_id"]).first()
        if topic is None:
            raise ValueError(f"Topic with id {state['topic_id']} not found")
        location = topic.name

    response = (
        services.get_llm(conf)
        .with_structured_output(WeatherQueryClassification)
        .invoke(CLASSIFICATION_PROMPT_TEMPLATE.format(query=query, location=location))
    )

    # workaround for https://github.com/langchain-ai/langchain/discussions/28853
    assert isinstance(response, WeatherQueryClassification)

    external_knowledge_sources: list[ExternalKnowledgeSource] = []
    if response.is_weather_related:
        external_knowledge_sources.append({"type": "weather", "location": location})

    return {
        "query": query,
        "external_knowledge_sources": external_knowledge_sources,
    }


CLASSIFICATION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template=(
        "Analyze the following query and topic and determine if it's asking about the weather at a given location. "
        "Respond in JSON format with 'is_weather_related' as a boolean and 'location' as a string.\n\n"
        "Location: {location}\n"
        "Query: '{query}'\n"
    ),
)


class WeatherQueryClassification(BaseModel):
    is_weather_related: bool = Field(description="whether the query is weather-related")
    location: str | None = Field(description="location the query is about, if any")
