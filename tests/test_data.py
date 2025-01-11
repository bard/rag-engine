from src.data import parse_insurance_table, load_insurance_data_from_url


def test_parse_insurance_table(snapshot, sample_html):
    insurance_records = parse_insurance_table(sample_html)
    assert insurance_records == snapshot


def test_load_insurance_data_with_data_url(snapshot, sample_html_data_uri):
    documents = load_insurance_data_from_url(sample_html_data_uri)
    assert (len(documents)) == 10
    assert documents[0] == snapshot


def test_documents_have_stable_ids_based_on_year(sample_html_data_uri):
    documents = load_insurance_data_from_url(sample_html_data_uri)
    assert documents[0].id == "insurance-record-2012"
