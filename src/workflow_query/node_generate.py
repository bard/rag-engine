import pprint
from typing import TypedDict
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from ..config import Config
from .. import services
from .state import AgentState


class GenerateStateUpdate(TypedDict):
    messages: list[BaseMessage]
    sources: list[str]


def generate(state: AgentState, config: RunnableConfig) -> GenerateStateUpdate:
    """Generate answer."""
    c = Config.from_runnable_config(config)
    llm = services.get_llm(c)

    documents = state["documents"]
    query = state["query"]
    assert query is not None
    weather_info = state["weather_info"]
    location = state["location"]

    docs_content = "\n\n".join(doc.page_content for doc in documents)

    sources = [f"knowledge_base[{doc.id}]" for doc in documents]

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
    )
    if weather_info is not None and location is not None:
        sources.append("external[weather]")
        system_message_content += (
            f"Current weather information for {location}: {weather_info}\n\n"
        )

    system_message_content += f"{docs_content}\n\n"

    prompt = [SystemMessage(system_message_content), HumanMessage(query)]

    response = llm.invoke(prompt)
    return {"messages": [response], "sources": sources}
