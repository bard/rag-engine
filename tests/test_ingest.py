import pytest
from langchain_chroma.vectorstores import Chroma
from numpy import extract
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from langchain_core.runnables.config import RunnableConfig
from sqlalchemy import create_engine
from src.data import InsuranceRecord, SQLInsuranceRecord
from src.ingest import AgentState, SourceContent, fetch, ingest, extract, get_graph


def test_fetch_node(agent_config, sample_html_as_data_url, snapshot):
    agent_state = AgentState(
        url=sample_html_as_data_url, source_content=None, insurance_records=[]
    )

    new_agent_state = fetch(agent_state, agent_config)

    assert new_agent_state.get("source_content") == snapshot


def test_extract_node(agent_config, sample_html, sample_html_as_data_url, snapshot):
    agent_state = AgentState(
        url=sample_html_as_data_url,
        source_content=SourceContent(data=sample_html, type="text/html"),
        insurance_records=[],
    )

    new_agent_state = extract(agent_state, agent_config)

    assert new_agent_state.get("records") == snapshot


def test_ingest_node(agent_config, sample_html_as_data_url, snapshot):
    agent_state = AgentState(
        url=sample_html_as_data_url,
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
    )

    new_agent_state = ingest(agent_state, agent_config)

    database_state = (
        Session(create_engine(agent_config.get("configurable")["db_url"]))
        .query(SQLInsuranceRecord)
        .all()
    )
    vector_store_state = Chroma(
        collection_name="documents",
        persist_directory=agent_config.get("configurable")
        .get("vector_store")
        .get("path"),
    ).get()
    assert new_agent_state == snapshot
    assert database_state == snapshot
    assert vector_store_state == snapshot


def test_graph(agent_config, sample_html_as_data_url, snapshot):
    agent_state = AgentState(
        url=sample_html_as_data_url, source_content=None, insurance_records=[]
    )

    graph = get_graph()
    new_agent_state = graph.invoke(agent_state, config=agent_config)

    database_state = (
        Session(create_engine(agent_config.get("configurable")["db_url"]))
        .query(SQLInsuranceRecord)
        .all()
    )
    vector_store_state = Chroma(
        collection_name="documents",
        persist_directory=agent_config.get("configurable")
        .get("vector_store")
        .get("path"),
    ).get()
    assert new_agent_state == snapshot
    assert database_state == snapshot
    assert vector_store_state == snapshot


@pytest.fixture
def agent_config(tmp_path, tmp_db_url) -> RunnableConfig:
    return RunnableConfig(
        configurable={
            "openai_api_key": "fake_key",
            "db_url": tmp_db_url,
            "vector_store": {
                "type": "chroma",
                "collection_name": "documents",
                "path": f"{tmp_path}/chroma",
                "embeddings_type": "local-minilm",
            },
        }
    )
