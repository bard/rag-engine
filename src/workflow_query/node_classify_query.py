from typing import TypedDict
from langchain_core.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from .. import services
from ..config import Config
from ..workflow_query.state import AgentState, ExternalKnowledgeSource


class ClassifyStateUpdate(TypedDict):
    query: str
    location: str | None
    external_knowledge_sources: list[ExternalKnowledgeSource]


CLASSIFICATION_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["query"],
    template=(
        "Analyze the following query and determine if it's asking about the weather for a location. "
        "If it is, extract the location mentioned. "
        "Respond in JSON format with 'is_weather_related' as a boolean and 'location' as a string or null.\n\n"
        "Query: '{query}'"
    ),
)


class WeatherQueryClassification(BaseModel):
    is_weather_related: bool = Field(description="whether the query is weather-related")
    location: str | None = Field(description="location the query is about, if any")


def classify_query(state: AgentState, config: RunnableConfig) -> ClassifyStateUpdate:
    c = Config.from_runnable_config(config)

    last_message = state["messages"][-1]
    assert isinstance(last_message, HumanMessage)
    query = last_message.content
    assert isinstance(query, str)

    response = (
        services.get_llm(c)
        .with_structured_output(WeatherQueryClassification)
        .invoke(CLASSIFICATION_PROMPT_TEMPLATE.format(query=query))
    )

    # workaround for https://github.com/langchain-ai/langchain/discussions/28853
    assert isinstance(response, WeatherQueryClassification)

    external_knowledge_sources: list[ExternalKnowledgeSource] = []
    if response.is_weather_related:
        assert response.location is not None
        external_knowledge_sources.append(
            {"type": "weather", "location": response.location}
        )

    return {
        "location": response.location,
        "query": query,
        "external_knowledge_sources": external_knowledge_sources,
    }
