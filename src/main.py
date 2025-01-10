from langchain.schema import Document
from dotenv import load_dotenv
from .data import load_insurance_data_from_url

load_dotenv(override=True)


class DocumentProcessor:
    """A class for processing insurance data files into Document objects"""

    def load_data(self, url: str) -> list[Document]:
        return load_insurance_data_from_url(url)
