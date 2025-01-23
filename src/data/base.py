import hashlib
from typing import Self, Optional
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel


class IndexableData(BaseModel):
    title: Optional[str]
    source_url: str

    def id(self) -> str:
        content_hash = hashlib.sha256(self.to_text().encode("utf-8")).hexdigest()
        # TODO consider using a url-based id (allows replacing/updating as long as urls are stable)
        return f"{self.__class__.__name__}-{content_hash[:8]}"

    def to_text(self) -> str:
        raise Exception("to_text() not defined")

    @classmethod
    def from_content(
        cls,
        content_data: str,
        content_type: str,
        source_url: str,
        llm: BaseChatModel | None,
    ) -> Self | None:
        raise Exception("from_content() not defined")
