from typing import TypedDict
from langchain_core.runnables.config import RunnableConfig

from .. import services
from ..util import fetch_content
from ..config import Config
from .state import AgentState, SourceContent


class FetchStateUpdate(TypedDict):
    source_content: SourceContent


def fetch(state: AgentState, config: RunnableConfig) -> FetchStateUpdate:
    conf = Config.from_runnable_config(config)

    log = services.get_logger(conf)
    log.debug("node/fetch")

    result = fetch_content(state["url"])

    return {
        "source_content": SourceContent(data=result["data"], type=result["type"]),
    }
