from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from sqlalchemy.orm import Session

from .. import services
from ..data import SQLInsuranceRecord
from .state import AgentState
from ..config import AgentConfig


def index_and_store(state: AgentState, config: RunnableConfig) -> None:
    c = AgentConfig.from_runnable_config(config)
    url = state["url"]

    db = services.get_db(c)
    vector_store = services.get_vector_store(c)
    documents: list[Document] = []

    insurance_records = state["insurance_records"]
    with Session(db) as session:
        with session.begin():
            for r in insurance_records:
                session.add(SQLInsuranceRecord(**r.model_dump()))
                documents.append(r.to_langchain_document(source_url=url))

    generic_data = state["generic_data"]
    if generic_data is not None:
        documents.append(generic_data.to_langchain_document())

    vector_store.add_documents(documents)

    return None
