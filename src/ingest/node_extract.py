from .state import AgentState
from .config import RunnableConfig
from ..data import parse_insurance_table


def extract(state: AgentState, config: RunnableConfig) -> AgentState:
    content = state.get("source_content")
    assert content is not None

    insurance_records = parse_insurance_table(content.get("data"))
    return {**state, "insurance_records": insurance_records}
