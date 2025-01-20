import pytest
from langchain_chroma.vectorstores import Chroma
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from src.data import InsuranceRecord, SQLInsuranceRecord
from src.ingest import (
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
        insurance_records=[],
        generic_data=None,
    )

    state_update = fetch(agent_state, agent_config)

    assert state_update.get("source_content") == snapshot


def test_extract_insurance_records(
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
        insurance_records=[],
        generic_data=None,
    )

    state_update = extract(agent_state, agent_config)

    assert state_update.get("records") == snapshot


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
        insurance_records=[],
        generic_data=None,
    )

    state_update = extract(agent_state, agent_config)

    assert state_update.get("insurance_records") == []
    assert state_update.get("generic_data") == snapshot


def test_index_and_store(
    agent_config, average_insurance_expenditures_html_as_data_url, snapshot
):
    agent_state = AgentState(
        url=average_insurance_expenditures_html_as_data_url,
        source_content=None,
        insurance_records=[
            InsuranceRecord(
                year=2012,
                average_expenditure=812.4,
                percent_change=2.2,
                source_content="<tr>\n<td>2012</td>\n<td>$812.40</td>\n<td>2.2%</td>\n</tr>",
            ),
            InsuranceRecord(
                year=2013,
                average_expenditure=841.06,
                percent_change=3.5,
                source_content="<tr>\n<td>2013</td>\n<td>841.06</td>\n<td>3.5</td>\n</tr>",
            ),
        ],
        generic_data=None,
    )

    state_update = index_and_store(agent_state, agent_config)

    database_state = (
        Session(create_engine(agent_config.get("configurable").get("db").get("url")))
        .query(SQLInsuranceRecord)
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
        insurance_records=[],
        generic_data=None,
    )

    graph = get_graph()
    state_update = graph.invoke(agent_state, config=agent_config)

    database_state = (
        Session(create_engine(agent_config.get("configurable").get("db").get("url")))
        .query(SQLInsuranceRecord)
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
