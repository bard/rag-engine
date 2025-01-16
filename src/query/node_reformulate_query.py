from typing import Literal
from langchain_core.runnables.config import RunnableConfig

from ..services import get_llm
from ..config import AgentConfig
from .state import AgentState


def reformulate_query(
    state: AgentState, config: RunnableConfig
) -> dict[Literal["query"], str]:
    c = AgentConfig.from_runnable_config(config)

    llm = get_llm(c)
    user_message = state["messages"][-1].content

    prompt = f"Reformulate the following user question into a concise search query: '{user_message}'"
    response = llm.invoke(prompt)
    reformulated_query = str(response.content)

    return {"query": reformulated_query}
