import json
import pprint
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from sqlalchemy.orm import Session
from sqlalchemy import insert
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
                # TODO: since id's are content-ids, use on_conflict_do_nothing instead
                session.merge(
                    SqlDocument(
                        id=r.id(),
                        readable=r.to_readable(),
                        data=r.model_dump_json(),
                    )
                )

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=c.indexing.chunk_size,
                    chunk_overlap=c.indexing.chunk_overlap,
                ).split_text(r.to_readable())

                for i, chunk in enumerate(chunks):
                    doc_id = r.id()
                    chunk_id = i
                    doc = Document(
                        id=f"{doc_id}-{chunk_id}",
                        page_content=chunk,
                        metadata={
                            "source_id": doc_id,
                            "chunk_id": chunk_id,
                            "source_url": r.source_url,
                            "title": r.title,
                        },
                    )
                    documents.append(doc)

    vector_store.add_documents(documents)

    return None
