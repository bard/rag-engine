from langchain.schema import Document
from typing import List
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sqlalchemy.orm import Mapped, mapped_column, declarative_base


SqlAlchemyBase = declarative_base()


class InsuranceRecord(BaseModel):
    """Pydantic model for insurance record data"""

    year: int
    average_expenditure: float
    percent_change: float
    source_content: str


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


def parse_insurance_table(html_content: str) -> List[InsuranceRecord]:
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
        record = InsuranceRecord(
            year=year,
            average_expenditure=expenditure,
            percent_change=percent_change,
            source_content=str(row),
        )
        results.append(record)

    return results


def insurance_record_to_langchain_document(
    record: InsuranceRecord, source_url: str
) -> Document:
    """
    Convert an InsuranceRecord to a LangChain Document.

    Args:
        record: The insurance record to convert
        source_url: The URL where the record was sourced from

    Returns:
        Document object containing the record data and metadata
    """
    content = "\n".join(
        [
            f"Year: {record.year}, "
            f"Expenditure: ${record.average_expenditure:.2f}, "
            f"Change: {record.percent_change}%"
        ]
    )

    return Document(
        page_content=content,
        id=f"insurance-record-{record.year}",
        metadata={
            "source_url": source_url,
            "source_content": record.source_content,
            "year": record.year,
            "average_expenditure": record.average_expenditure,
            "percent_change": record.percent_change,
        },
    )
