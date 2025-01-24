from typing import TypedDict
from langchain_core.runnables.config import RunnableConfig

from .state import AgentState
from ..config import Config
from .. import services
from ..data import (
    IndexableData,
    InsuranceAverageExpenditureData,
    GenericTabularData,
    TextualData,
)


class ExtractStateUpdate(TypedDict):
    extracted_data: list[IndexableData]


EXTRACTORS: list[type[IndexableData]] = [
    InsuranceAverageExpenditureData,
    GenericTabularData,
    TextualData,
]


def extract(state: AgentState, config: RunnableConfig) -> ExtractStateUpdate:
    conf = Config.from_runnable_config(config)

    log = services.get_logger(conf)
    log.debug("node/extract")

    url = state["url"]
    content = state["source_content"]
    assert content is not None

    extracted_data: list[IndexableData] = []
    llm = services.get_llm(conf)

    # naive example of deciding between a cheap/specialized/fast
    # strategy to extract data from known source type, a
    # expensive/generic/slow best-effort fallback for structured but
    # otherwise unknown data, and a final fallback for purerly textual data
    # TODO: make extractors configurable at a system level
    for extractor in EXTRACTORS:
        result = extractor.from_content(
            content_data=content["data"],
            source_url=url,
            content_type=content["type"],
            llm=llm,
        )
        if result is not None:
            extracted_data.append(result)
            break

    if not extracted_data:
        log.warning(f"No data could be extracted from {url}")

    return {
        "extracted_data": extracted_data,
    }
