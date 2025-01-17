from src.data import parse_insurance_table, insurance_record_to_langchain_document


def test_parse_insurance_table(snapshot, average_insurance_expenditures_html):
    insurance_records = parse_insurance_table(average_insurance_expenditures_html)
    assert insurance_records == snapshot


def test_insurance_record_to_langchain_document(
    snapshot, average_insurance_expenditures_html
):
    insurance_records = parse_insurance_table(average_insurance_expenditures_html)
    document = insurance_record_to_langchain_document(
        insurance_records[0], source_url="about:blank"
    )
    assert document == snapshot


def test_documents_have_generated_id_based_on_year(average_insurance_expenditures_html):
    insurance_records = parse_insurance_table(average_insurance_expenditures_html)
    document = insurance_record_to_langchain_document(
        insurance_records[0], source_url="about:blank"
    )
    assert document.id == "insurance-record-2012"
