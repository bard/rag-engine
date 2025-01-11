from typing import List
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import SecretStr
from .data import load_insurance_data_from_url
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class Settings(BaseSettings):
    pinecone_api_key: SecretStr
    openai_api_key: SecretStr


class DocumentProcessor:
    """A class for processing insurance data files into Document objects"""

    def load_data(self, url: str) -> list[Document]:
        return load_insurance_data_from_url(url)


class VectorStore:
    def __init__(
        self,
        pinecone_api_key: SecretStr,
        pinecone_index_name: str,
        openai_api_key: SecretStr,
    ):
        """Initialize the vector store with OpenAI embeddings and Pinecone"""
        embeddings = OpenAIEmbeddings(api_key=openai_api_key)
        self.vector_store = PineconeVectorStore(
            index_name=pinecone_index_name,
            pinecone_api_key=pinecone_api_key.get_secret_value(),
            embedding=embeddings,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        self.vector_store.add_documents(documents)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform similarity search using the vector store.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k)
