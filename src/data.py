from langchain.schema import Document
from typing import List
from pydantic import BaseModel
from bs4 import BeautifulSoup
import requests
import base64
from urllib.parse import urlparse


class InsuranceRecord(BaseModel):
    """Pydantic model for insurance record data"""

    year: int
    average_expenditure: float
    percent_change: float
    source_content: str


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


def fetch_html_content(source_url: str) -> str:
    """
    Fetch HTML content from various sources: local file, HTTP URL, or data URL.

    Args:
        source_url: file URL, HTTP URL, or data URL containing HTML content

    Returns:
        HTML content as string

    Raises:
        ValueError: If the source type is not supported or content cannot be retrieved
    """
    parsed = urlparse(str(source_url))

    # Handle local file
    if not parsed.scheme or parsed.scheme == "file":
        with open(parsed.path, "r") as file:
            return file.read()

    # Handle HTTP(S) URLs
    if parsed.scheme in ["http", "https"]:
        response = requests.get(str(source_url))
        response.raise_for_status()
        return response.text

    # Handle data URLs
    if parsed.scheme == "data":
        # Split metadata and content
        header, encoded = source_url.split(",", 1)
        if ";base64" in header:
            return base64.b64decode(encoded).decode("utf-8")
        return encoded

    raise ValueError(f"Unsupported source type: {parsed.scheme}")


def load_insurance_data_from_url(source_url: str) -> list[Document]:
    """
    Load insurance data file and convert to Document objects with parsed insurance records.
    Supports HTML files containing insurance expenditure tables.

    Args:
        file_path: Path to the insurance data file

    Returns:
        List of Document objects containing parsed insurance records with metadata
    """
    # Fetch and parse HTML content
    html_content = fetch_html_content(source_url)
    records = parse_insurance_table(html_content)
    documents = []
    for r in records:
        content = "\n".join(
            [
                f"Year: {r.year}, "
                f"Expenditure: ${r.average_expenditure:.2f}, "
                f"Change: {r.percent_change}%"
            ]
        )
        doc = Document(
            page_content=content,
            id=f"insurance-record-{r.year}",
            metadata={
                "source": source_url,
                "source_content": r.source_content,
            },
        )
        documents.append(doc)

    return documents
