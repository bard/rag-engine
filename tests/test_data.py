import pytest
from src.data import GenericTabularData, TextualData


@pytest.mark.xfail(reason="todo")
def test_generic_tabular_data_from_content(snapshot):
    pass


def test_textual_data_from_html(snapshot):
    textual_data = TextualData.from_content(
        source_url="about:blank",
        content_data="<html><title>foobar</title><body><p>Hello, world!</p></body></html>",
        content_type="text/html",
        llm=None,
    )

    assert textual_data is not None
    assert textual_data.to_text() == snapshot


def test_textual_data_from_text(snapshot):
    textual_data = TextualData.from_content(
        source_url="about:blank",
        content_data="Hello, world!",
        content_type="text/html",
        llm=None,
    )

    assert textual_data is not None
    assert textual_data.to_text() == snapshot
