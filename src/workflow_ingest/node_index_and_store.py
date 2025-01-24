import pprint
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import services
from ..db import SqlKnowledgeBaseDocument
from ..config import Config
from .state import AgentState


def index_and_store(state: AgentState, config: RunnableConfig) -> None:
    conf = Config.from_runnable_config(config)

    log = services.get_logger(conf)
    log.debug("node/index_and_store")

    db = services.get_db(conf)
    vector_store = services.get_vector_store(conf)
    documents: list[Document] = []

    topic_id = state["topic_id"]
    extracted_data = state["extracted_data"]
    with Session(db) as session:
        with session.begin():
            for r in extracted_data:
                # TODO: since hash-based id's reliably represent the
                # document, use on_conflict_do_nothing instead to
                # avoid the useless update
                session.merge(
                    SqlKnowledgeBaseDocument(
                        id=r.id(),
                        content=r.to_text(),
                        data=r.model_dump_json(),
                        topic_id=topic_id,
                    )
                )

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=conf.indexing.chunk_size,
                    chunk_overlap=conf.indexing.chunk_overlap,
                ).split_text(r.to_text())

                for i, chunk in enumerate(chunks):
                    doc_id = r.id()
                    chunk_id = i
                    metadata = {
                        "source_id": doc_id,
                        "chunk_id": chunk_id,
                        "source_url": r.source_url,
                    }
                    if r.title is not None:
                        metadata["title"] = r.title

                    if state["topic_id"] is None:
                        metadata["topic_id"] = "UNCATEGORIZED"
                    else:
                        metadata["topic_id"] = state["topic_id"]

                    documents.append(
                        Document(
                            id=f"{doc_id}-{chunk_id}",
                            page_content=chunk,
                            metadata=metadata,
                        )
                    )

    vector_store.add_documents(documents)

    return None
