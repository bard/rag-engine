import pprint
from datetime import datetime, timezone
import pytest
from langchain.schema import HumanMessage, Document
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session
from src.db import SqlTopic

from src.workflow_query import (
    AgentState,
    get_graph,
    retrieve_from_knowledge_base,
    classify_query,
)
from src import services
from src.workflow_query.node_generate import generate


@pytest.mark.vcr
def test_classify_query_detects_weather_query(config):
    with Session(services.get_db(config)) as session:
        paris_topic = SqlTopic(name="Paris", created_at=datetime.now(timezone.utc))
        session.add(paris_topic)
        session.commit()
        session.refresh(paris_topic)

    agent_state = AgentState(
        messages=[HumanMessage(content="how is the weather?")],
        retrieved_knowledge=[],
        query=None,
        topic_id=paris_topic.id,
        external_knowledge_sources=[],
    )

    state_update = classify_query(agent_state, config.to_runnable_config())

    assert state_update == {
        "query": "how is the weather?",
        "external_knowledge_sources": [{"type": "weather", "location": "Paris"}],
    }


@pytest.mark.vcr
def test_classify_query_determines_query_not_to_be_about_weather_if_no_topic_provided(
    config,
):
    agent_state = AgentState(
        messages=[HumanMessage(content="how is the weather?")],
        retrieved_knowledge=[],
        query=None,
        topic_id=None,
        external_knowledge_sources=[],
    )

    state_update = classify_query(agent_state, config.to_runnable_config())

    assert state_update == {
        "query": "how is the weather?",
        "external_knowledge_sources": [],
    }


def test_retrieve_with_travel_knowledge_base(
    config, travel_knowledge_documents, snapshot
):
    vector_store = services.get_vector_store(config)
    vector_store.add_documents(travel_knowledge_documents)

    agent_state = AgentState(
        messages=[HumanMessage(content="what are some nice things to see in Paris?")],
        retrieved_knowledge=[],
        query="what are some nice things to see in Paris?",
        topic_id="Paris",
        external_knowledge_sources=[],
    )

    state_update = retrieve_from_knowledge_base(
        agent_state, config.to_runnable_config()
    )

    assert state_update["retrieved_knowledge"] == snapshot


@pytest.mark.vcr
def test_generate_from_knowledge_base(config, travel_knowledge_documents, snapshot):
    agent_state = AgentState(
        messages=[HumanMessage(content="what are some nice things to see in Paris?")],
        retrieved_knowledge=travel_knowledge_documents[0:1],
        query="what are some nice things to see in Paris?",
        topic_id=None,
        external_knowledge_sources=[],
    )

    state_update = generate(agent_state, config.to_runnable_config())

    assert state_update["messages"][-1].content == snapshot


@pytest.mark.vcr
def test_generate_from_external_service(config, snapshot):
    agent_state = AgentState(
        messages=[HumanMessage(content="what is the weather like in paris?")],
        retrieved_knowledge=[
            Document(
                page_content="Current weather information for paris: Temperature: -0.55°C, Feels like: -3.75°C, Humidity: 95%, Status: mist",
                id="external[weather]",
            )
        ],
        query="what is the weather like in paris?",
        topic_id="aaaa-bbbb",
        external_knowledge_sources=[],
    )

    state_update = generate(agent_state, config.to_runnable_config())

    assert state_update["messages"][-1].content == snapshot


@pytest.mark.vcr
def test_graph_with_weather_query(config, travel_knowledge_documents, snapshot):
    with Session(services.get_db(config)) as session:
        paris_topic = SqlTopic(name="Paris", created_at=datetime.now(timezone.utc))
        session.add(paris_topic)
        session.commit()
        session.refresh(paris_topic)

    services.get_vector_store(config).add_documents(travel_knowledge_documents)

    agent_state = AgentState(
        messages=[HumanMessage(content="what is the weather like?")],
        retrieved_knowledge=[],
        query=None,
        topic_id=paris_topic.id,
        external_knowledge_sources=[],
    )

    response = get_graph().invoke(agent_state, config.to_runnable_config())

    assert response.get("messages", [])[-1].content == snapshot


@pytest.mark.vcr
def test_graph_with_knowledge_query(config, travel_knowledge_documents, snapshot):
    with Session(services.get_db(config)) as session:
        paris_topic = SqlTopic(name="Paris", created_at=datetime.now(timezone.utc))
        session.add(paris_topic)
        session.commit()
        session.refresh(paris_topic)

    vector_store = services.get_vector_store(config)
    vector_store.add_documents(travel_knowledge_documents)
    agent_state = AgentState(
        messages=[HumanMessage(content="what are some nice things to see?")],
        retrieved_knowledge=[],
        query=None,
        topic_id=paris_topic.id,
        external_knowledge_sources=[],
    )

    response = get_graph().invoke(agent_state, config.to_runnable_config())

    assert response.get("messages", [])[-1].content == snapshot
