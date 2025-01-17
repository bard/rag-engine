from src.data import GenericTabularData, InsuranceRecord


def test_parse_insurance_table(snapshot, average_insurance_expenditures_html):
    insurance_records = InsuranceRecord.from_html_content(
        average_insurance_expenditures_html
    )
    assert insurance_records == snapshot


def test_get_langchain_document_from_insurance_record(
    snapshot, average_insurance_expenditures_html
):
    insurance_records = InsuranceRecord.from_html_content(
        average_insurance_expenditures_html
    )
    document = insurance_records[0].to_langchain_document(source_url="about:blank")
    assert document == snapshot


def test_insurance_record_documents_have_generated_id_based_on_year(
    average_insurance_expenditures_html,
):
    insurance_records = InsuranceRecord.from_html_content(
        average_insurance_expenditures_html
    )
    document = insurance_records[0].to_langchain_document(source_url="about:blank")
    assert document.id == "insurance-record-2012"


def test_get_langchain_document_from_generic_data(
    snapshot,
):
    generic_data = GenericTabularData(
        title="Some Data",
        data=[{"foo": 1, "bar": 2}, {"foo": 4, "bar": 5}],
        content_hash="abc123",
    )

    document = generic_data.to_langchain_document()

    assert document == snapshot
