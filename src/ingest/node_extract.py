from typing import TypedDict
from langchain_core.runnables.config import RunnableConfig

from .state import AgentState
from ..data import InsuranceRecord, parse_insurance_table


class ExtractStateUpdate(TypedDict):
    insurance_records: list[InsuranceRecord]


def extract(state: AgentState, config: RunnableConfig) -> ExtractStateUpdate:
    content = state.get("source_content")
    assert content is not None

    insurance_records = parse_insurance_table(content.get("data"))

    return {"insurance_records": insurance_records}
