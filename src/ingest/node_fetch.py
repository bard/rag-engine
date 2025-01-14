from .state import AgentState, SourceContent
from .config import RunnableConfig
from ..util import fetch_html_content


def fetch(state: AgentState, config: RunnableConfig) -> AgentState:
    url = state.get("url")
    # TODO: support more file types (e.g. excel)
    html_content = fetch_html_content(url)

    return {
        **state,
        "source_content": SourceContent(data=html_content, type="text/html"),
    }
