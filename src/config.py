from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field
from typing import Self, Union, Literal


class PineconeVectorStoreBackend(BaseModel):
    type: Literal["pinecone"]
    index_name: str
    api_key: str
    embeddings_type: Literal["openai"]
    score_threshold: float = 0.1


class MockVectorStoreBackend(BaseModel):
    type: Literal["mock"]
    score_threshold: float = 0.1


class ChromaVectorStoreBackend(BaseModel):
    type: Literal["chroma"]
    path: str
    collection_name: str
    embeddings_type: Literal["local-minilm", "openai"]
    score_threshold: float = 0.1


type VectorStoreBackend = Union[
    PineconeVectorStoreBackend, MockVectorStoreBackend, ChromaVectorStoreBackend
]


class OpenaiLlmBackend(BaseModel):
    type: Literal["openai"]
    model: str
    api_key: str


class AnthropicLlmBackend(BaseModel):
    type: Literal["anthropic"]
    model: str
    api_key: str


class DbBackend(BaseModel):
    url: str


type LlmBackend = Union[OpenaiLlmBackend, AnthropicLlmBackend]


class AgentConfig(BaseModel):
    db: DbBackend
    llm: LlmBackend = Field(discriminator="type")
    vector_store: VectorStoreBackend = Field(discriminator="type")

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> Self:
        """Validate runnable config"""
        configurable = config.get("configurable")
        assert configurable is not None
        return cls(**configurable)
