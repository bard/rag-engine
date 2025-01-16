import pprint
from typing import TypedDict
from langchain.schema import BaseMessage, Document, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import AgentConfig
from ..services import get_llm
from .state import AgentState


class GenerateStateUpdate(TypedDict):
    messages: list[BaseMessage]


def generate(state: AgentState, config: RunnableConfig) -> GenerateStateUpdate:
    """Generate answer."""
    c = AgentConfig.from_runnable_config(config)
    llm = get_llm(c)

    documents = state.get("documents")
    assert documents is not None
    query = state.get("query")
    assert query is not None
    weather_info = state.get("weather_info")
    location = state.get("location")

    docs_content = "\n\n".join(doc.page_content for doc in documents)

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
    )
    if weather_info is not None and location is not None:
        system_message_content += (
            f"Current weather information for {location}: {weather_info}\n\n"
        )

    system_message_content += f"{docs_content}\n\n"

    prompt = [SystemMessage(system_message_content), HumanMessage(query)]

    response = llm.invoke(prompt)
    return {"messages": [response]}
