
from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from .agents.document_processor import DocumentProcessor
from .agents.vector_store import VectorStore

load_dotenv(override=True)


class Settings(BaseSettings):
    pinecone_api_key: SecretStr
    openai_api_key: SecretStr
