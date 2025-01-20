import pytest
import pprint
from src.data import GenericTabularData, ExpenditureReport


def test_parse_average_expenditures_document(
    snapshot, average_insurance_expenditures_html
):
    expenditure_record = ExpenditureReport.from_html_content(
        average_insurance_expenditures_html, source_url="about:blank"
    )

    assert expenditure_record == snapshot


def test_get_langchain_document_from_expenditure_report(
    snapshot, average_insurance_expenditures_html
):
    expenditure_report = ExpenditureReport.from_html_content(
        average_insurance_expenditures_html, source_url="about:blank"
    )

    document = expenditure_report.to_langchain_document()
    assert document == snapshot


def test_get_langchain_document_from_generic_data(
    snapshot,
):
    generic_data = GenericTabularData(
        title="Some Data",
        data=[{"foo": 1, "bar": 2}, {"foo": 4, "bar": 5}],
        source_url="about:blank",
    )

    document = generic_data.to_langchain_document()

    assert document == snapshot
