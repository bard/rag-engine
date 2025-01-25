import pprint
from typing import TypedDict
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from sqlalchemy.orm import Session

from ..config import Config
from .. import services
from ..db import SqlTopic
from .state import AgentState


class GenerateStateUpdate(TypedDict):
    messages: list[BaseMessage]


def generate(state: AgentState, config: RunnableConfig) -> GenerateStateUpdate:
    conf = Config.from_runnable_config(config)

    query = state["query"]
    assert query is not None

    system_message_content = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
    )

    if state["topic_id"]:
        db = services.get_db(conf)
        with Session(db) as session:
            topic = (
                session.query(SqlTopic).filter(SqlTopic.id == state["topic_id"]).first()
            )
            if topic:
                system_message_content += f"Topic: {topic}\n\n"

    system_message_content += "\n\n".join(
        doc.page_content for doc in state["retrieved_knowledge"]
    )

    prompt = [SystemMessage(system_message_content), HumanMessage(query)]

    response = services.get_llm(conf).invoke(prompt)

    return {"messages": [response]}
