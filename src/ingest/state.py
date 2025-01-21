from typing import TypedDict, Literal
from ..data import ExpenditureReport, GenericReport


class SourceContent(TypedDict):
    data: str
    type: Literal["text/html"]  # TODO 'application/vnd.ms-excel']]


class AgentState(TypedDict):
    url: str
    source_content: SourceContent | None
    extracted_data: list[ExpenditureReport | GenericReport]
