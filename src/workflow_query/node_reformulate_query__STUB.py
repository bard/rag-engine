from typing import Literal
from langchain_core.runnables.config import RunnableConfig

from .. import services
from ..config import Config
from .state import AgentState


def reformulate_query(
    state: AgentState, config: RunnableConfig
) -> dict[Literal["query"], str]:
    conf = Config.from_runnable_config(config)

    llm = services.get_llm(conf)
    user_message = state["messages"][-1].content

    prompt = f"Reformulate the following user question into a concise search query: '{user_message}'"
    response = llm.invoke(prompt)
    query = str(response.content)

    return {"query": query}
