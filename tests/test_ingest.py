import pprint
import pytest
from langchain_chroma.vectorstores import Chroma
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from src.data.textual import TextualData
from src.db import SqlKnowledgeBaseDocument
from src.workflow_ingest import (
    AgentState,
    SourceContent,
    fetch,
    extract,
    index_and_store,
    get_graph,
)


def test_fetch(config, travel_knowledge_data_as_data_url, snapshot):
    agent_state = AgentState(
        url=travel_knowledge_data_as_data_url,
        source_content=None,
        extracted_data=[],
        topic_id=None,
    )

    state_update = fetch(agent_state, config.to_runnable_config())

    assert state_update.get("source_content") == snapshot


@pytest.mark.vcr
@pytest.mark.timeout(60)  # longer timeout for LLM processing
def test_extract_generic_tabular_data(
    config,
    premiums_by_state_html,
    premiums_by_state_html_as_data_url,
    snapshot,
):
    agent_state = AgentState(
        url=premiums_by_state_html_as_data_url,
        source_content=SourceContent(data=premiums_by_state_html, type="text/html"),
        extracted_data=[],
        topic_id=None,
    )

    state_update = extract(agent_state, config.to_runnable_config())

    assert state_update["extracted_data"] == snapshot


def test_extract_textual_data(config, snapshot):
    source_content = SourceContent(
        data="<h1>lorem ipsum</h1><p>Vivamus id enim.  Aenean in sem ac leo mollis blandit.</p>",
        type="text/html",
    )

    agent_state = AgentState(
        url="about:blank",
        source_content=source_content,
        extracted_data=[],
        topic_id=None,
    )

    state_update = extract(agent_state, config.to_runnable_config())
    assert state_update == snapshot


def test_index_and_store(
    config,
    snapshot,
):
    data = TextualData.from_content(
        content_data="Paris, the 'City of Light,' boasts iconic landmarks such as the Eiffel Tower, offering panoramic views from its observation decks. The Louvre Museum, home to masterpieces like the Mona Lisa, attracts art enthusiasts worldwide. Notre-Dame Cathedral, a Gothic architectural marvel, stands on the Île de la Cité. The Champs-Élysées, lined with shops and cafes, leads to the Arc de Triomphe, honoring those who fought for France. Montmartre, with its artistic heritage, features the Basilica of the Sacré-Cœur atop its hill.",
        content_type="text/plain",
        source_url="about:blank",
        llm=None,
    )
    assert data is not None

    agent_state = AgentState(
        url="about:blank",
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


def test_graph(config, travel_knowledge_data_as_data_url, snapshot):
    agent_state = AgentState(
        url=travel_knowledge_data_as_data_url,
        topic_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        source_content=None,
        extracted_data=[],
    )

    graph = get_graph()
    result = graph.invoke(agent_state, config.to_runnable_config())

    database_state = (
        Session(create_engine(config.db.url)).query(SqlKnowledgeBaseDocument).all()
    )
    vector_store_state = Chroma(
        collection_name="documents", persist_directory=config.vector_store.path
    ).get()

    assert result == snapshot
    assert database_state == snapshot
    assert vector_store_state == snapshot
