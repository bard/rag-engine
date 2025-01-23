from typing import Self
from langchain_core.language_models import BaseChatModel
from bs4 import BeautifulSoup

from .base import IndexableData


class TextualData(IndexableData):
    data: str

    @classmethod
    def from_content(
        cls,
        content_data: str,
        content_type: str,
        source_url: str,
        llm: BaseChatModel | None,
    ) -> Self | None:
        if content_type == "text/plain":
            return cls(title=source_url, data=content_data, source_url=source_url)
        elif content_type == "text/html":
            soup = BeautifulSoup(content_data, "html.parser")

            # Try to find title in order of preference: <title>, <h1>, URL
            title_tag = soup.find("title")
            h1_tag = soup.find("h1")

            if title_tag and title_tag.get_text(strip=True):
                title = title_tag.get_text(strip=True)
            elif h1_tag and h1_tag.get_text(strip=True):
                title = h1_tag.get_text(strip=True)
            else:
                title = None

            # Get text content, preserving some structture
            converted_text = soup.get_text(separator="\n", strip=True)

            return cls(title=title, source_url=source_url, data=converted_text)
        else:
            return None

    def to_text(self) -> str:
        text = ""
        if self.title is not None:
            text += self.title + "\n\n"
        text += self.data
        return text
