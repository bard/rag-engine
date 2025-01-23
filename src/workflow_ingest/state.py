from typing import TypedDict, Literal
from ..data import IndexableData


class SourceContent(TypedDict):
    data: str
    type: Literal["text/html", "text/plain"]  # TODO 'application/vnd.ms-excel'


class AgentState(TypedDict):
    url: str
    topic_id: str | None
    source_content: SourceContent | None
    extracted_data: list[IndexableData]
