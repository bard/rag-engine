import logging
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
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr
from pyowm import OWM  # type: ignore[import-untyped]
from .config import Config

logger = logging.getLogger()


def get_db(config: Config) -> Engine:
    return create_engine(config.db.url)


def get_logger(config: Config) -> logging.Logger:
    return logger


def get_weather_client(config: Config) -> OWM:
    return OWM(config.weather.api_key)


def get_llm(config: Config) -> BaseChatModel:
    if config.llm.type == "openai":
        return ChatOpenAI(model=config.llm.model)
    else:
        raise Exception(f"Unsupported LLM backend: {config.llm.type}")


def get_vector_store(config: Config) -> VectorStore:
    if config.vector_store.type == "chroma":
        embedding_function: ChromaEmbeddingsAdapter | OpenAIEmbeddings
        if config.embeddings.type == "chroma-internal":
            default_chroma_embedding_function = ChromaDefaultEmbeddingFunction()
            assert default_chroma_embedding_function is not None
            embedding_function = ChromaEmbeddingsAdapter(
                default_chroma_embedding_function
            )
        elif config.embeddings.type == "openai":
            embedding_function = OpenAIEmbeddings(
                api_key=SecretStr(config.embeddings.api_key)
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
