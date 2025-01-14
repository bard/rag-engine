from typing import TypedDict, Literal
from ..data import InsuranceRecord


class SourceContent(TypedDict):
    data: str
    # TODO Union[Literal['text/html'], Literal['application/vnd.ms-excel']]
    type: Literal["text/html"]


class AgentState(TypedDict):
    url: str
    source_content: SourceContent | None
    insurance_records: list[InsuranceRecord]
