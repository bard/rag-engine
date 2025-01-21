from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field
from typing import Self, Union, Literal


class PineconeVectorStoreConfig(BaseModel):
    type: Literal["pinecone"]
    index_name: str
    api_key: str
    score_threshold: float = 0.1


class ChromaVectorStoreConfig(BaseModel):
    type: Literal["chroma"]
    path: str
    collection_name: str
    score_threshold: float = 0.1


class OpenaiEmbeddingsConfig(BaseModel):
    type: Literal["openai"]
    api_key: str


class ChromaInternalEmbeddingsConfig(BaseModel):
    type: Literal["chroma-internal"]


class IndexingConfig(BaseModel):
    chunk_size: int
    chunk_overlap: int


class OpenaiLlmConfig(BaseModel):
    type: Literal["openai"]
    model: str
    api_key: str


class AnthropicLlmConfig(BaseModel):
    type: Literal["anthropic"]
    model: str
    api_key: str


class DbConfig(BaseModel):
    url: str


class OpenWeatherMapConfig(BaseModel):
    api_key: str


class Config(BaseModel):
    db: DbConfig
    llm: Union[OpenaiLlmConfig, AnthropicLlmConfig] = Field(discriminator="type")
    embeddings: Union[OpenaiEmbeddingsConfig, ChromaInternalEmbeddingsConfig] = Field(
        discriminator="type"
    )
    vector_store: Union[PineconeVectorStoreConfig, ChromaVectorStoreConfig] = Field(
        discriminator="type"
    )
    indexing: IndexingConfig = IndexingConfig(chunk_size=1000, chunk_overlap=100)
    weather: OpenWeatherMapConfig

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Self:
        """Validate runnable config"""
        configurable = config.get("configurable")
        assert configurable is not None
        return cls(**configurable)
