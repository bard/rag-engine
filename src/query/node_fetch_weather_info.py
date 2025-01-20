from langchain_core.runnables.config import RunnableConfig
from .. import services
from ..config import AgentConfig
from ..query.state import AgentState


def fetch_weather_info(state: AgentState, config: RunnableConfig):
    c = AgentConfig.from_runnable_config(config)
    weather = services.get_weather_client(c)

    location = state.get("location")
    assert location is not None

    observation = weather.weather_manager().weather_at_place(location)
    # TODO: handle case in which no observation is available
    assert observation is not None

    weather_data = observation.weather
    weather_info = (
        f"Temperature: {weather_data.temperature('celsius')['temp']}°C, "
        f"Feels like: {weather_data.temperature('celsius')['feels_like']}°C, "
        f"Humidity: {weather_data.humidity}%, "
        f"Status: {weather_data.detailed_status}"
    )

    return {"weather_info": weather_info}
