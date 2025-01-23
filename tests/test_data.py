import pytest
from src.data import InsuranceAverageExpenditureData, GenericTabularData, TextualData


def test_expenditure_data_from_content(snapshot, average_insurance_expenditures_html):
    expenditure_data = InsuranceAverageExpenditureData.from_content(
        content_data=average_insurance_expenditures_html,
        content_type="text/html",
        source_url="about:blank",
        llm=None,
    )

    assert expenditure_data is not None
    assert expenditure_data == snapshot
    assert expenditure_data.to_text() == snapshot


def test_generic_tabular_data_to_text(snapshot):
    generic_data = GenericTabularData(
        title="Title", source_url="about:blank", data=[{"foo": "bar"}]
    )

    assert generic_data.to_text() == snapshot


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
