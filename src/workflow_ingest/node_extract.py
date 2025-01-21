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
        result = extractor.from_html(html, url, llm)
        if result is not None:
            extracted_data.append(result)
            break

    if not extracted_data:
        log.warning(f"No data could be extracted from {url}")

    return {
        "extracted_data": extracted_data,
    }


class GenericTabularDataExtraction(BaseModel):
    title: str
    data: list[Dict[str, Any]]


def parse_generic_tabular_data_with_llm(
    html: str, source_url: str, llm: BaseChatModel
) -> GenericTabularData | None:
    try:
        # `with_structured_output` consistently causes the
        # `data` field to not be generated, so we fall back to
        # old-style output parser
        parser = PydanticOutputParser(pydantic_object=GenericTabularDataExtraction)
        response = llm.invoke(
            [
                SystemMessage(f"""You are an assistant for data extraction tasks. Extract a title
                    and tabular data from the provided html.
                    Provide the output in JSON format: {parser.get_format_instructions()}.`"""),
                HumanMessage(html),
            ]
        )
        raw_generic_data = parser.parse(str(response.content))

        return GenericTabularData(
            source_url=source_url,
            **raw_generic_data.model_dump(),
        )

    except:  # TODO log errors
        return None
