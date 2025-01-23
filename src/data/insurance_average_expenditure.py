from typing import Self
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from bs4 import BeautifulSoup

from .base import IndexableData


class InsuranceAverageExpenditureData(IndexableData):
    class InsuranceRecord(BaseModel):
        year: int
        average_expenditure: float
        percent_change: float

    data: list[InsuranceRecord]

    def to_text(self) -> str:
        md = f"# {self.title}\n\n"
        for kv_pair in self.data:
            line = ", ".join([f"{k}: {v}" for k, v in kv_pair])
            line += "\n\n"
            md += line

        return md

    @classmethod
    def from_content(
        cls,
        content_data: str,
        content_type: str,
        source_url: str,
        llm: BaseChatModel | None,
    ) -> Self | None:
        """
        Parse the auto insurance expenditure table and return a list of dictionaries.
        Each dictionary represents a row with year, expenditure, and percent change.
        """

        if content_type != "text/html":
            return None
        if "Average Expenditures for Auto Insurance" not in content_data:
            return None

        # Create BeautifulSoup object
        soup = BeautifulSoup(content_data, "html.parser")

        # Find the innermost table with the actual data
        tables = soup.find_all("table")

        title_table = tables[1]
        title_td = title_table.find("td")
        title = title_td.get_text(strip=True)

        data_table = tables[2]

        # Initialize list to store results
        data: list[InsuranceAverageExpenditureData.InsuranceRecord] = []

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
            record = InsuranceAverageExpenditureData.InsuranceRecord(
                year=year,
                average_expenditure=expenditure,
                percent_change=percent_change,
            )
            data.append(record)

        return cls(title=title, data=data, source_url=source_url)
