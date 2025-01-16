import chromadb.utils.embedding_functions as ef
from langchain_core.embeddings import Embeddings
from chromadb.api.types import EmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from langchain_core.runnables.config import RunnableConfig
from langchain.schema import Document
from pydantic import SecretStr
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from ..data import (
    SQLInsuranceRecord,
    insurance_record_to_langchain_document,
)
from .state import AgentState
from .config import AgentConfig, validate_agent_config


def ingest(state: AgentState, config: RunnableConfig) -> AgentState:
    c = validate_agent_config(config)
    url = state.get("url")

    engine = create_engine(c.db_url)
    vector_store = get_vector_store(c)

    insurance_records = state.get("insurance_records")
    documents: list[Document] = []

    with Session(engine) as session:
        with session.begin():
            for r in insurance_records:
                session.add(SQLInsuranceRecord(**r.model_dump()))
                documents.append(
                    insurance_record_to_langchain_document(r, source_url=url)
                )

    vector_store.add_documents(documents)

    # TODO don't return anything as there is no state update
    return state


def get_vector_store(config: AgentConfig) -> VectorStore:
    if config.vector_store.type == "chroma":
        if config.vector_store.embeddings_type == "local-minilm":
            default_chroma_embedding_function = ef.DefaultEmbeddingFunction()
            if default_chroma_embedding_function is None:
                raise Exception("Default embedding function for Chroma is None")
            embedding_function = ChromaEmbeddingsAdapter(
                default_chroma_embedding_function
            )

        elif config.vector_store.embeddings_type == "openai":
            embedding_function = OpenAIEmbeddings(
                api_key=SecretStr(config.openai_api_key)
            )
        else:
            raise Exception(f"Unknown embedding type: {config.vector_store.type}")

        return Chroma(
            collection_name=config.vector_store.collection_name,
            embedding_function=embedding_function,
            persist_directory=config.vector_store.path,
        )
    else:
        raise Exception("boom")


def get_db(config: AgentConfig) -> Engine:
    return create_engine(config.db_url)


# https://cookbook.chromadb.dev/integrations/langchain/embeddings/#custom-adapter
class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):  # pyright: ignore
        return self.ef(texts)

    def embed_query(self, query):  # pyright: ignore
        return self.ef([query])[0]
