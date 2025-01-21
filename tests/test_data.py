from src.data import ExpenditureReport


def test_parse_average_expenditures_document(
    snapshot, average_insurance_expenditures_html
):
    expenditure_record = ExpenditureReport.from_html_content(
        average_insurance_expenditures_html, source_url="about:blank"
    )

    assert expenditure_record == snapshot
