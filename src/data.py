import hashlib
from typing import Dict, Any, Self
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import inspect


class BaseIndexedContent(BaseModel):
    title: str
    source_url: str
    data: list[Any]

    def id(self) -> str:
        content_hash = hashlib.sha256(
            (self.title + "".join(str(item) for item in self.data)).encode("utf-8")
        ).hexdigest()

        # TODO evaluate url-based id (allows replacing/updating as long as urls are stable)
        return f"{self.__class__.__name__}-{content_hash[:8]}"

    def to_readable(self) -> str:
        md = f"# {self.title}\n\n"

        if isinstance(self.data, BaseModel):
            kv_pairs = self.data.model_dump()
        else:
            kv_pairs = self.data

        for kv_pair in kv_pairs:
            line = ", ".join([f"{k}: {v}" for k, v in kv_pair])
            line += "\n\n"
            md += line
        return md


class GenericReport(BaseIndexedContent):
    data: list[Dict[str, Any]]


class InsuranceRecord(BaseModel):
    """Pydantic model for insurance record data"""

    year: int
    average_expenditure: float
    percent_change: float


class ExpenditureReport(BaseIndexedContent):
    data: list[InsuranceRecord]

    @classmethod
    def from_html_content(cls, html_content: str, source_url: str) -> Self:
        """
        Parse the auto insurance expenditure table and return a list of dictionaries.
        Each dictionary represents a row with year, expenditure, and percent change.
        """
        # Create BeautifulSoup object
        soup = BeautifulSoup(html_content, "html.parser")

        # Find the innermost table with the actual data
        tables = soup.find_all("table")

        title_table = tables[1]
        title_td = title_table.find("td")
        title = title_td.get_text(strip=True)

        data_table = tables[2]

        # Initialize list to store results
        data: list[InsuranceRecord] = []

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
            )
            data.append(record)

        return cls(title=title, data=data, source_url=source_url)


class SqlAlchemyBase(DeclarativeBase):
    pass


class SqlDocument(SqlAlchemyBase):
    """SqlAlchemy model for stored document"""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(primary_key=True)
    data: Mapped[str] = mapped_column(nullable=False)
    readable: Mapped[str] = mapped_column(nullable=False)
    # TODO add title and source url fields

    def __repr__(self) -> str:
        return f"SqlDocument(id={self.id}, data={self.data}, readable={self.readable}"

    def to_dict(self):
        return {c.key: getattr(self, c.key) for c in inspect(self).mapper.column_attrs}
