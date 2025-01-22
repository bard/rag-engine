from typing import Dict, Any, Self
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain.schema import HumanMessage, SystemMessage
from pydantic import BaseModel


from .base import IndexableData


class GenericTabularData(IndexableData):
    data: list[Dict[str, Any]]

    def to_text(self) -> str:
        md = f"# {self.title}\n\n"
        for kv_pair in self.data:
            line = ", ".join([f"{k}: {v}" for k, v in kv_pair.items()])
            line += "\n\n"
            md += line

        return md

    @classmethod
    def from_html(
        cls, html: str, source_url: str, llm: BaseChatModel | None
    ) -> Self | None:
        assert llm is not None

        if "<table>" not in html:
            return None

        class GenericTabularDataExtraction(BaseModel):
            title: str
            data: list[Dict[str, Any]]

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

            return cls(
                source_url=source_url,
                **raw_generic_data.model_dump(),
            )

        except:  # TODO log errors
            return None
