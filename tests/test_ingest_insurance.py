import pprint
import pytest
from langchain_chroma.vectorstores import Chroma
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from src.db import SqlKnowledgeBaseDocument
from src.data import InsuranceAverageExpenditureData
from src.workflow_ingest import (
    AgentState,
    index_and_store,
)


def test_index_and_store(
    config,
    average_insurance_expenditures_html_as_data_url,
    average_insurance_expenditures_html,
    snapshot,
):
    data = InsuranceAverageExpenditureData.from_content(
        content_data=average_insurance_expenditures_html,
        content_type="text/html",
        source_url=average_insurance_expenditures_html_as_data_url,
        llm=None,
    )
    assert data is not None

    agent_state = AgentState(
        url=average_insurance_expenditures_html_as_data_url,
        topic_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        source_content=None,
        extracted_data=[data],
    )

    state_update = index_and_store(agent_state, config.to_runnable_config())
    database_state = (
        Session(create_engine(config.db.url)).query(SqlKnowledgeBaseDocument).all()
    )
    vector_store_state = Chroma(
        collection_name="documents", persist_directory=config.vector_store.path
    ).get()

    assert state_update == snapshot
    assert database_state == snapshot
    assert vector_store_state == snapshot
