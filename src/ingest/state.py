from typing import TypedDict, Literal
from ..data import GenericTabularData, InsuranceRecord


class SourceContent(TypedDict):
    data: str
    type: Literal["text/html"]  # TODO 'application/vnd.ms-excel']]


class AgentState(TypedDict):
    url: str
    source_content: SourceContent | None
    insurance_records: list[InsuranceRecord]
    generic_data: GenericTabularData | None
