import hashlib
from typing import Sequence, TypedDict
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.config import RunnableConfig

from .state import AgentState
from ..config import AgentConfig
from ..data import (
    ExpenditureReport,
    InsuranceRecord,
    RawGenericTabularData,
    GenericTabularData,
)
from .. import services


class ExtractStateUpdate(TypedDict):
    extracted_data: list[ExpenditureReport | GenericTabularData]


def extract(state: AgentState, config: RunnableConfig) -> ExtractStateUpdate:
    c = AgentConfig.from_runnable_config(config)

    log = services.get_logger(c)
    log.debug("node/extract")

    url = state["url"]
    content = state["source_content"]
    assert content is not None

    # naive example of deciding between a cheap/specialized/fast
    # strategy to extract data from known sources and a
    # expensive/generic/slow best-effort fallback for unknown data.

    html = content["data"]
    extracted_data: list[ExpenditureReport | GenericTabularData] = []
    if "Average Expenditures for Auto Insurance" in html:
        generic_data = None
        extracted_data.append(ExpenditureReport.from_html_content(html, source_url=url))
    else:
        generic_data = parse_generic_tabular_data_with_llm(html, c, source_url=url)
        if generic_data is not None:
            extracted_data.append(generic_data)

    # TODO log a warning if no data was extracted

    return {
        "extracted_data": extracted_data,
    }


def parse_generic_tabular_data_with_llm(
    html: str, config: AgentConfig, source_url: str
) -> GenericTabularData | None:
    try:
        # `with_structured_output` consistently causes the
        # `data` field to not be generated, so we fall back to
        # old-style output parser
        parser = PydanticOutputParser(pydantic_object=RawGenericTabularData)
        response = services.get_llm(config).invoke(
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
