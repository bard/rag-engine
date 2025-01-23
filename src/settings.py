from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from typing import Literal


load_dotenv(override=True)


class Settings(BaseSettings):
    cors_origins: str = "http://localhost:3000,http://localhost:8000,http://localhost:5173"
    pinecone_api_key: SecretStr
    openai_api_key: SecretStr
    openweathermap_api_key: SecretStr
    db_url: SecretStr
    chroma_db_path: str
    log_level: Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] = (
        "INFO"
    )
