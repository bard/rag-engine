import pprint
from langchain_core.runnables.config import RunnableConfig
from src.config import AgentConfig
from src.query.state import AgentState


def fetch_weather_info(state: AgentState, config: RunnableConfig):
    c = AgentConfig.from_runnable_config(config)
    # TODO weather_api_client = get_weather_api_client(c)

    location = state.get("location")
    assert location is not None

    weather_info_stub = "min: 5°C min; max: 10°C; sunny."

    return {"weather_info": weather_info_stub}
