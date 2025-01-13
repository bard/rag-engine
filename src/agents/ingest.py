import chromadb.utils.embedding_functions as ef
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from langchain_core.runnables.config import RunnableConfig
from typing import TypedDict
from langchain.schema import Document
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session
from ..data import (
    SQLInsuranceRecord,
    insurance_record_to_langchain_document,
    parse_insurance_table,
)
from ..util import fetch_html_content
from typing import Union, Literal


class PineconeVectorStoreBackend(BaseModel):
    type: Literal["pinecone"]
    index_name: str
    api_key: str
    embeddings_type: Literal["openai"]


class MockVectorStoreBackend(BaseModel):
    type: Literal["mock"]


class ChromaVectorStoreBackend(BaseModel):
    type: Literal["chroma"]
    path: str
    collection_name: str
    embeddings_type: Union[Literal["local-minilm"], Literal["openai"]]


class AgentConfig(BaseModel):
    openai_api_key: str
    db_url: str
    vector_store: Union[
        PineconeVectorStoreBackend, MockVectorStoreBackend, ChromaVectorStoreBackend
    ] = Field(discriminator="type")


class AgentState(TypedDict):
    url: str


def validate_agent_config(config: RunnableConfig) -> AgentConfig:
    """Validate agent config"""
    configurable = config.get("configurable")
    if configurable is None:
        raise Exception("Missing agent configuration")
    return AgentConfig(**configurable)


def ingest(state: AgentState, _config: RunnableConfig) -> AgentState:
    config = validate_agent_config(_config)

    url = state.get("url")
    if url is None:
        raise Exception("missing state")

    engine = create_engine(config.db_url)
    vector_store = get_vector_store(config)

    html_content = fetch_html_content(url)
    insurance_records = parse_insurance_table(html_content)
    documents: list[Document] = []

    with Session(engine) as session:
        with session.begin():
            for r in insurance_records:
                session.add(SQLInsuranceRecord(**r.model_dump()))
                documents.append(
                    insurance_record_to_langchain_document(r, source_url=url)
                )

    vector_store.add_documents(documents)

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


from langchain_core.embeddings import Embeddings
from chromadb.api.types import EmbeddingFunction


# https://cookbook.chromadb.dev/integrations/langchain/embeddings/#custom-adapter
class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):  # pyright: ignore
        return self.ef(texts)

    def embed_query(self, query):  # pyright: ignore
        return self.ef([query])[0]
