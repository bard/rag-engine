import pprint
from typing import Union
from langchain_core.prompts.prompt import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.config import AgentConfig
from src.query.state import AgentState
from src.services import get_llm


def classify_query(state: AgentState, config: RunnableConfig):
    c = AgentConfig.from_runnable_config(config)
    last_message = state.get("messages")[-1]
    assert isinstance(last_message, HumanMessage)
    query = last_message.content

    response = (
        get_llm(c)
        .with_structured_output(WeatherQueryClassification)
        .invoke(classification_prompt_template.format(query=query))
    )

    # workaround for https://github.com/langchain-ai/langchain/discussions/28853
    assert isinstance(response, WeatherQueryClassification)

    return {
        "is_weather_query": response.is_weather_related,
        "location": response.location,
        "query": query,
    }


classification_prompt_template = PromptTemplate(
    input_variables=["query"],
    template=(
        "Analyze the following query and determine if it's asking about the weather for a location. "
        "If it is, extract the location mentioned. "
        "Respond in JSON format with 'is_weather_related' as a boolean and 'location' as a string.\n\n"
        "Query: '{query}'"
    ),
)


class WeatherQueryClassification(BaseModel):
    is_weather_related: bool = Field(
        description="whether the query has to do with weather"
    )
    location: Union[None, str] = Field(
        description="location the query is about, if it's weather-related"
    )
