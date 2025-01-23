from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel
from typing import TypedDict, Dict, Any
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.config import RunnableConfig

from .state import AgentState
from ..config import Config
from ..data import (
    IndexableData,
    InsuranceAverageExpenditureData,
    GenericTabularData,
    TextualData,
)
from .. import services


class ExtractStateUpdate(TypedDict):
    extracted_data: list[IndexableData]


def extract(state: AgentState, config: RunnableConfig) -> ExtractStateUpdate:
    c = Config.from_runnable_config(config)

    log = services.get_logger(c)
    log.debug("node/extract")

    url = state["url"]
    content = state["source_content"]
    assert content is not None

    html = content["data"]
    extracted_data: list[IndexableData] = []
    llm = services.get_llm(c)

    # naive example of deciding between a cheap/specialized/fast
    # strategy to extract data from known source type, a
    # expensive/generic/slow best-effort fallback for structured but
    # otherwise unknown data, and a final fallback for purerly textual data
    # TODO: make extractors configurable at a system level
    extractors: list[type[IndexableData]] = [
        InsuranceAverageExpenditureData,
        GenericTabularData,
        TextualData,
    ]

    for extractor in extractors:
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
