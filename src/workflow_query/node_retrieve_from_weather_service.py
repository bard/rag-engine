from typing import TypedDict
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from .. import services
from ..config import Config
from ..workflow_query.state import AgentState


class FetchWeatherInfoStateUpdate(TypedDict):
    retrieved_knowledge: list[Document]


def retrieve_from_weather_service(
    state: AgentState, config: RunnableConfig
) -> FetchWeatherInfoStateUpdate:
    conf = Config.from_runnable_config(config)

    weather_query = next(
        (
            source
            for source in state["external_knowledge_sources"]
            if source["type"] == "weather"
        ),
        None,
    )
    if weather_query is None:
        raise Exception("No weather query found")

    location = weather_query["location"]

    weather = services.get_weather_client(conf)
    observation = weather.weather_manager().weather_at_place(location)

    # TODO: handle case in which no observation is available
    assert observation is not None

    weather_data = observation.weather
    weather_info = (
        f"Current weather information for {location}: "
        f"Temperature: {weather_data.temperature('celsius')['temp']}°C, "
        f"Feels like: {weather_data.temperature('celsius')['feels_like']}°C, "
        f"Humidity: {weather_data.humidity}%, "
        f"Status: {weather_data.detailed_status}"
    )

    return {
        "retrieved_knowledge": [
            *state["retrieved_knowledge"],
            Document(page_content=weather_info, id="weather-info"),
        ]
    }
