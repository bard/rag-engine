import hashlib
from typing import TypedDict
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables.config import RunnableConfig

from .state import AgentState
from ..config import AgentConfig
from ..data import InsuranceRecord, RawGenericTabularData, GenericTabularData
from .. import services


class ExtractStateUpdate(TypedDict):
    insurance_records: list[InsuranceRecord]
    generic_data: GenericTabularData | None


def extract(state: AgentState, config: RunnableConfig) -> ExtractStateUpdate:
    c = AgentConfig.from_runnable_config(config)

    log = services.get_logger(c)
    log.debug("node/extract")

    content = state.get("source_content")
    assert content is not None

    # naive example of deciding between a cheap/specialized/fast
    # strategy to extract data from known sources and a
    # expensive/generic/slow best-effort fallback for unknown data.

    html = content["data"]
    if "Average Expenditures for Auto Insurance" in html:
        insurance_records = InsuranceRecord.from_html_content(html)
        generic_data = None
    else:
        insurance_records = []
        generic_data = parse_generic_tabular_data_with_llm(html, c)

    # provide a warning if no data at all was extracted

    return {"insurance_records": insurance_records, "generic_data": generic_data}


def parse_generic_tabular_data_with_llm(
    html: str, config: AgentConfig
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

        content_hash = hashlib.sha256(
            (
                raw_generic_data.title + "".join(str(item) for item in raw_generic_data)
            ).encode("utf-8")
        ).hexdigest()

        generic_data = GenericTabularData(
            **raw_generic_data.model_dump(), content_hash=content_hash
        )

        return generic_data
    except:
        return None
