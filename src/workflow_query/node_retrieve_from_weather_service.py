from typing import TypedDict
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from .. import services
from ..config import Config
from ..workflow_query.state import AgentState


class FetchWeatherInfoStateUpdate(TypedDict):
    documents: list[Document]


def retrieve_from_weather_service(
    state: AgentState, config: RunnableConfig
) -> FetchWeatherInfoStateUpdate:
    c = Config.from_runnable_config(config)
    weather = services.get_weather_client(c)

    location = state["location"]
    assert location is not None

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
        "documents": [
            *state["documents"],
            Document(page_content=weather_info, id="weather-info"),
        ]
    }
