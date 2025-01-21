from typing import TypedDict, Literal
from ..data import IndexableData


class SourceContent(TypedDict):
    data: str
    type: Literal["text/html"]  # TODO 'application/vnd.ms-excel', 'text/plain'


class AgentState(TypedDict):
    url: str
    source_content: SourceContent | None
    extracted_data: list[IndexableData]
