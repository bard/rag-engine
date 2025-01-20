from typing import TypedDict

from langchain_core.runnables.config import RunnableConfig
from .state import AgentState, SourceContent
from ..util import fetch_html_content


class FetchStateUpdate(TypedDict):
    source_content: SourceContent


def fetch(state: AgentState, config: RunnableConfig) -> FetchStateUpdate:
    url = state["url"]
    # TODO: support more file types (e.g. excel)
    html_content = fetch_html_content(url)

    return {
        "source_content": SourceContent(data=html_content, type="text/html"),
    }
