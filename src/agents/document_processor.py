from langchain.schema import Document
from sqlalchemy.orm import Session
from ..data import (
    SQLInsuranceRecord,
    insurance_record_to_langchain_document,
    parse_insurance_table,
)
from ..util import fetch_html_content


class DocumentProcessor:
    """A class for processing insurance data files into Document objects"""

    db: Session

    def __init__(self, db: Session):
        self.db = db

    def load_data(self, url: str) -> list[Document]:
        """
        Load insurance data from a URL and store it in the database.
        The data is sourced from the Insurance Information Institute (III)
        at https://www.iii.org/table-archive/20916

        Args:
            url (str): The URL to fetch insurance data from

        Returns:
            list[Document]: List of processed Document objects
        """

        html_content = fetch_html_content(url)
        insurance_records = parse_insurance_table(html_content)

        documents: list[Document] = []
        with self.db.begin():
            for r in insurance_records:
                self.db.add(SQLInsuranceRecord(**r.model_dump()))
                documents.append(
                    insurance_record_to_langchain_document(r, source_url=url)
                )

        return documents
