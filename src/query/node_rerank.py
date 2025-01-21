from langchain_core.runnables import RunnableConfig

from .state import AgentState


def rerank(state: AgentState, config: RunnableConfig) -> None:
    "[STUB] use LLM to prioritize documents based on semantic relevance to the query."
    pass
