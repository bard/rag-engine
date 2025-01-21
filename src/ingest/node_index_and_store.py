import json
import pprint
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from sqlalchemy.orm import Session

from .. import services
from ..data import SqlDocument
from .state import AgentState
from ..config import AgentConfig


def index_and_store(state: AgentState, config: RunnableConfig) -> None:
    c = AgentConfig.from_runnable_config(config)

    log = services.get_logger(c)
    log.debug("node/index_and_store")

    db = services.get_db(c)
    vector_store = services.get_vector_store(c)
    documents: list[Document] = []

    reports = state["extracted_data"]
    with Session(db) as session:
        with session.begin():
            for r in reports:
                session.add(
                    SqlDocument(
                        id=r.id(),
                        readable=r.to_readable(),
                        data=r.model_dump_json(),
                    )
                )
                # split the report's readable representation as returned by to_readable() into multiple langchain documents using a textsplitter, and append them to the documents array ai!

    vector_store.add_documents(documents)

    return None
