import pprint
import pytest
from langchain_chroma.vectorstores import Chroma
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from src.sql import SqlKnowledgeBaseDocument
from src.data import InsuranceAverageExpenditureData
from src.workflow_ingest import (
    AgentState,
    SourceContent,
    fetch,
    extract,
    index_and_store,
    get_graph,
)


def test_fetch(agent_config, average_insurance_expenditures_html_as_data_url, snapshot):
    agent_state = AgentState(
        url=average_insurance_expenditures_html_as_data_url,
        source_content=None,
        extracted_data=[],
    )

    state_update = fetch(agent_state, agent_config)

    assert state_update.get("source_content") == snapshot


def test_extract_expenditure_data(
    agent_config,
    average_insurance_expenditures_html,
    average_insurance_expenditures_html_as_data_url,
    snapshot,
):
    agent_state = AgentState(
        url=average_insurance_expenditures_html_as_data_url,
        source_content=SourceContent(
            data=average_insurance_expenditures_html, type="text/html"
        ),
        extracted_data=[],
    )

    state_update = extract(agent_state, agent_config)
    assert state_update["extracted_data"] == snapshot


@pytest.mark.vcr
@pytest.mark.timeout(60)  # longer timeout for LLM processing
def test_extract_generic_tabular_data(
    agent_config,
    premiums_by_state_html,
    premiums_by_state_html_as_data_url,
    snapshot,
):
    agent_state = AgentState(
        url=premiums_by_state_html_as_data_url,
        source_content=SourceContent(data=premiums_by_state_html, type="text/html"),
        extracted_data=[],
    )

    state_update = extract(agent_state, agent_config)

    assert state_update["extracted_data"] == snapshot


def test_extract_textual_data(agent_config, snapshot):
    source_content = SourceContent(
        data="<h1>lorem ipsum</h1><p>Vivamus id enim.  Aenean in sem ac leo mollis blandit.</p>",
        type="text/html",
    )

    agent_state = AgentState(
        url="about:blank",
        source_content=source_content,
        extracted_data=[],
    )

    state_update = extract(agent_state, agent_config)
    assert state_update == snapshot


def test_index_and_store(
    agent_config,
    average_insurance_expenditures_html_as_data_url,
    average_insurance_expenditures_html,
    snapshot,
):
    agent_state = AgentState(
        url=average_insurance_expenditures_html_as_data_url,
        source_content=None,
        extracted_data=[
            InsuranceAverageExpenditureData.from_html(
                average_insurance_expenditures_html,
                source_url=average_insurance_expenditures_html_as_data_url,
                llm=None,
            )
        ],
    )

    state_update = index_and_store(agent_state, agent_config)
    database_state = (
        Session(create_engine(agent_config.get("configurable").get("db").get("url")))
        .query(SqlKnowledgeBaseDocument)
        .all()
    )
    vector_store_state = Chroma(
        collection_name="documents",
        persist_directory=agent_config.get("configurable")
        .get("vector_store")
        .get("path"),
    ).get()

    assert state_update == snapshot
    assert database_state == snapshot
    assert vector_store_state == snapshot


def test_graph(agent_config, average_insurance_expenditures_html_as_data_url, snapshot):
    agent_state = AgentState(
        url=average_insurance_expenditures_html_as_data_url,
        source_content=None,
        extracted_data=[],
    )

    graph = get_graph()
    state_update = graph.invoke(agent_state, config=agent_config)

    database_state = (
        Session(create_engine(agent_config.get("configurable").get("db").get("url")))
        .query(SqlKnowledgeBaseDocument)
        .all()
    )
    vector_store_state = Chroma(
        collection_name="documents",
        persist_directory=agent_config.get("configurable")
        .get("vector_store")
        .get("path"),
    ).get()

    assert state_update == snapshot
    assert database_state == snapshot
    assert vector_store_state == snapshot
