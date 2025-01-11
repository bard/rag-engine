import base64
from src.data import parse_insurance_table, load_insurance_data_from_url


def test_parse_insurance_table(snapshot, sample_html):
    result = parse_insurance_table(sample_html)
    assert result == snapshot


def test_load_insurance_data_with_data_url(snapshot, sample_html):
    encoded_html = base64.b64encode(sample_html.encode("utf-8")).decode("utf-8")
    data_url = f"data:text/html;base64,{encoded_html}"
    documents = load_insurance_data_from_url(data_url)
    assert (len(documents)) == 10
    assert documents[0] == snapshot
