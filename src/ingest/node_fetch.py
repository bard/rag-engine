from typing import TypedDict
from langchain_core.runnables.config import RunnableConfig

from .. import services
from ..util import fetch_html_content
from ..config import AgentConfig
from .state import AgentState, SourceContent


class FetchStateUpdate(TypedDict):
    source_content: SourceContent


def fetch(state: AgentState, config: RunnableConfig) -> FetchStateUpdate:
    c = AgentConfig.from_runnable_config(config)

    log = services.get_logger(c)
    log.debug("node/fetch")

    url = state["url"]
    # TODO: support more file types (e.g. excel)
    html_content = fetch_html_content(url)

    return {
        "source_content": SourceContent(data=html_content, type="text/html"),
    }
