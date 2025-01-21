import pytest
from src.data import InsuranceAverageExpenditureData, GenericTabularData, TextualData


def test_expenditure_data_from_html(snapshot, average_insurance_expenditures_html):
    expenditure_data = InsuranceAverageExpenditureData.from_html(
        average_insurance_expenditures_html, source_url="about:blank", llm=None
    )

    assert expenditure_data == snapshot
    assert expenditure_data.to_readable() == snapshot


def test_generic_tabular_data_to_readable(snapshot):
    generic_data = GenericTabularData(
        title="Title", source_url="about:blank", data=[{"foo": "bar"}]
    )

    assert generic_data.to_readable() == snapshot


@pytest.mark.xfail(reason="todo")
def test_generic_tabular_data_from_html(snapshot):
    pass


def test_textual_data_to_readable(snapshot):
    textual_data = TextualData.from_html(
        source_url="about:blank",
        html="<html><title>foobar</title><body><p>Hello, world!</p></body></html>",
        llm=None,
    )

    assert textual_data.to_readable() == snapshot
