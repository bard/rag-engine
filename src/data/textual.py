from typing import Self
from langchain_core.language_models import BaseChatModel
from bs4 import BeautifulSoup

from .base import IndexableData


class TextualData(IndexableData):
    data: str

    @classmethod
    def from_html(
        cls, html: str, source_url: str, llm: BaseChatModel | None
    ) -> Self | None:
        soup = BeautifulSoup(html, "html.parser")

        # Try to find title in order of preference: <title>, <h1>, URL
        title_tag = soup.find("title")
        h1_tag = soup.find("h1")

        if title_tag and title_tag.get_text(strip=True):
            title = title_tag.get_text(strip=True)
        elif h1_tag and h1_tag.get_text(strip=True):
            title = h1_tag.get_text(strip=True)
        else:
            title = source_url

        # Get text content, preserving some structture
        converted_text = soup.get_text(separator="\n", strip=True)

        return cls(title=title, source_url=source_url, data=converted_text)

    def to_readable(self) -> str:
        return self.title + "\n\n" + self.data
