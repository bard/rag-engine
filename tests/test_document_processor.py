from src.data import SQLInsuranceRecord
from src.agents.document_processor import DocumentProcessor


def test_document_processor_db_interactions(
    average_insurance_expenditures_html_as_data_url, mock_db
):
    # Create document processor instance
    processor = DocumentProcessor(mock_db)

    # Load data using the processor
    documents = processor.load_data(average_insurance_expenditures_html_as_data_url)

    # Verify database interactions
    assert mock_db.add.call_count == len(documents)
    mock_db.begin.assert_called_once()


def test_document_processor_return_value(
    average_insurance_expenditures_html_as_data_url, mock_db
):
    # Create document processor instance
    processor = DocumentProcessor(mock_db)

    # Load data using the processor
    documents = processor.load_data(average_insurance_expenditures_html_as_data_url)

    # Verify documents were returned
    assert len(documents) > 0
    assert all(hasattr(doc, "page_content") for doc in documents)
    assert all(
        doc.metadata["source_url"] == average_insurance_expenditures_html_as_data_url
        for doc in documents
    )
    assert documents[0].id == "insurance-record-2012"
    assert documents[0].metadata["year"] == 2012
    assert documents[0].metadata["average_expenditure"] == 812.40
    assert documents[0].metadata["percent_change"] == 2.2
    assert (
        documents[0].metadata["source_content"]
        == """<tr>
<td>2012</td>
<td>$812.40</td>
<td>2.2%</td>
</tr>"""
    )


def test_document_processor_database_state(
    average_insurance_expenditures_html_as_data_url, sqlite_session, snapshot
):
    processor = DocumentProcessor(sqlite_session)

    processor.load_data(average_insurance_expenditures_html_as_data_url)
    all_records = sqlite_session.query(SQLInsuranceRecord).all()

    assert all_records == snapshot
