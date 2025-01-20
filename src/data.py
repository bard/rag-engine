from langchain.schema import Document
from typing import Dict, Any, Self
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase


class InsuranceRecord(BaseModel):
    """Pydantic model for insurance record data"""

    year: int
    average_expenditure: float
    percent_change: float
    source_content: str

    def to_langchain_document(self, source_url: str) -> Document:
        """
        Convert this InsuranceRecord to a LangChain Document.

        Args:
            source_url: The URL where the record was sourced from

        Returns:
            Document object containing the record data and metadata
        """
        content = "\n".join(
            [
                f"Year: {self.year}, "
                f"Expenditure: ${self.average_expenditure:.2f}, "
                f"Change: {self.percent_change}%"
            ]
        )

        return Document(
            page_content=content,
            id=f"insurance-record-{self.year}",
            metadata={
                "source_url": source_url,
                "source_content": self.source_content,
                "year": self.year,
                "average_expenditure": self.average_expenditure,
                "percent_change": self.percent_change,
            },
        )

    @classmethod
    def from_html_content(cls, html_content: str) -> list[Self]:
        """
        Parse the auto insurance expenditure table and return a list of dictionaries.
        Each dictionary represents a row with year, expenditure, and percent change.
        """
        # Create BeautifulSoup object
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the innermost table with the actual data
        tables = soup.find_all("table")
        data_table = tables[2]  # The third table contains our data

        # Initialize list to store results
        results = []

        # Process each row in tbody
        for row in data_table.tbody.find_all("tr"):
            cols = row.find_all("td")

            # Extract values
            year = int(cols[0].text.strip())

            # Clean and convert expenditure to float
            expenditure = cols[1].text.strip()
            expenditure = expenditure.replace("$", "").replace(",", "")
            expenditure = float(expenditure)

            # Clean and convert percent change to float
            percent_change = cols[2].text.strip()
            percent_change = percent_change.replace("%", "")
            percent_change = float(percent_change)

            # Create dictionary for current row
            record = cls(
                year=year,
                average_expenditure=expenditure,
                percent_change=percent_change,
                source_content=str(row),
            )
            results.append(record)

        return results


class SqlAlchemyBase(DeclarativeBase):
    pass


class SQLInsuranceRecord(SqlAlchemyBase):
    """SQLAlchemy model for insurance record data"""

    __tablename__ = "insurance_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    year: Mapped[int] = mapped_column(nullable=False)
    average_expenditure: Mapped[float] = mapped_column(nullable=False)
    percent_change: Mapped[float] = mapped_column(nullable=False)
    source_content: Mapped[str] = mapped_column(nullable=False)

    def __repr__(self) -> str:
        return (
            f"SQLInsuranceRecord(id={self.id}, "
            f"year={self.year}, "
            f"average_expenditure={self.average_expenditure:.2f}, "
            f"percent_change={self.percent_change})"
        )


class RawGenericTabularData(BaseModel):
    title: str
    data: list[Dict[str, Any]]


class GenericTabularData(RawGenericTabularData):
    content_hash: str

    def to_langchain_document(self) -> Document:
        """
        Convert this GenericTabularData to a LangChain Document.

        Returns:
            Document object containing the data and metadata
        """
        content = f"Title: {self.title}\n"
        for item in self.data:
            content += "\n".join(f"{k}: {v}" for k, v in item.items())
            content += "\n\n"

        return Document(
            page_content=content,
            id=f"generic-data-{self.content_hash[:8]}",
            metadata={
                "title": self.title,
                "content_hash": self.content_hash,
                "raw_data": self.data,
            },
        )
