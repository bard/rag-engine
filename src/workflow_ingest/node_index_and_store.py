from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from sqlalchemy.orm import Session
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .. import services
from ..db import SqlKnowledgeBaseDocument
from ..config import Config
from .state import AgentState


def index_and_store(state: AgentState, config: RunnableConfig) -> None:
    c = Config.from_runnable_config(config)

    log = services.get_logger(c)
    log.debug("node/index_and_store")

    db = services.get_db(c)
    vector_store = services.get_vector_store(c)
    documents: list[Document] = []

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
                    )
                )

                chunks = RecursiveCharacterTextSplitter(
                    chunk_size=c.indexing.chunk_size,
                    chunk_overlap=c.indexing.chunk_overlap,
                ).split_text(r.to_text())

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
