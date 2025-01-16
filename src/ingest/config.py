from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field
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
    embeddings_type: Literal["local-minilm", "openai"]


class AgentConfig(BaseModel):
    openai_api_key: str
    db_url: str
    vector_store: Union[
        PineconeVectorStoreBackend, MockVectorStoreBackend, ChromaVectorStoreBackend
    ] = Field(discriminator="type")


def validate_agent_config(config: RunnableConfig) -> AgentConfig:
    """Validate agent config"""
    configurable = config.get("configurable")
    if configurable is None:
        raise Exception("Missing agent configuration")
    return AgentConfig(**configurable)
