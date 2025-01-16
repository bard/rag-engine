from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction as ChromaDefaultEmbeddingFunction,
)
from sqlalchemy import Engine, create_engine
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings
from chromadb.api.types import EmbeddingFunction
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import VectorStore
from pydantic import SecretStr

from .config import AgentConfig


def get_db(config: AgentConfig) -> Engine:
    return create_engine(config.db.url)


def get_llm(config: AgentConfig) -> ChatOpenAI:
    if config.llm.type == "openai":
        return ChatOpenAI(model=config.llm.model, temperature=0)
    else:
        raise Exception(f"Unsupported LLM backend: {config.llm.type}")


def get_vector_store(config: AgentConfig) -> VectorStore:
    if config.vector_store.type == "chroma":
        if config.vector_store.embeddings_type == "local-minilm":
            default_chroma_embedding_function = ChromaDefaultEmbeddingFunction()
            assert default_chroma_embedding_function is not None
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
            # Prevents negative scores and consequent UserWarning. See https://github.com/langchain-ai/langchain/issues/10864
            collection_metadata={"hnsw:space": "cosine"},
        )
    else:
        raise Exception(f"Not implemented: {config.vector_store.type}")


# https://cookbook.chromadb.dev/integrations/langchain/embeddings/#custom-adapter
class ChromaEmbeddingsAdapter(Embeddings):
    def __init__(self, ef: EmbeddingFunction):
        self.ef = ef

    def embed_documents(self, texts):  # pyright: ignore
        return self.ef(texts)

    def embed_query(self, query):  # pyright: ignore
        return self.ef([query])[0]
