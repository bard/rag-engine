import pprint
import pytest
from langchain.schema import HumanMessage, Document
from langchain_core.messages import HumanMessage

from src.workflow_query import (
    AgentState,
    get_graph,
    retrieve_from_knowledge_base,
    reformulate_query,
    classify_query,
)
from src import services
from src.workflow_query.node_generate import generate


@pytest.mark.vcr
def test_classify_query(config):
    agent_state = AgentState(
        messages=[HumanMessage(content="how is the weather in Paris?")],
        documents=[],
        query=None,
        location=None,
        external_knowledge_sources=[],
    )

    state_update = classify_query(agent_state, config.to_runnable_config())

    assert state_update == {
        "location": "Paris",
        "query": "how is the weather in Paris?",
        "external_knowledge_sources": [{"type": "weather", "location": "Paris"}],
    }


def test_retrieve_with_travel_knowledge_base(
    config, travel_knowledge_documents, snapshot
):
    vector_store = services.get_vector_store(config)
    vector_store.add_documents(travel_knowledge_documents)

    agent_state = AgentState(
        messages=[HumanMessage(content="what are some nice things to see in Paris?")],
        documents=[],
        query="what are some nice things to see in Paris?",
        location=None,
        external_knowledge_sources=[],
    )

    state_update = retrieve_from_knowledge_base(
        agent_state, config.to_runnable_config()
    )

    assert state_update.get("documents") == snapshot


def test_retrieve_with_insurance_knowledge_base(config):
    vector_store = services.get_vector_store(config)
    vector_store.add_texts(
        [
            """# Average expenditures 2012-2013
Year: 2012, Expenditure: $812.40, Change: 2.2%
Year: 2013, Expenditure: $841.06, Change: 3.5%
""",
            """# Average expenditures 2014-2015
Year: 2014, Expenditure: $869.47, Change: 3.4%
Year: 2015, Expenditure: $896.66, Change: 3.1%
""",
            """# Average expenditures 2015-2016
Year: 2014, Expenditure: $869.47, Change: 3.4%
Year: 2015, Expenditure: $896.66, Change: 3.1%
""",
        ],
        metadatas=[],
    )
    agent_state = AgentState(
        messages=[
            HumanMessage(content="what is the expenditure in 2014?"),
        ],
        documents=[],
        query="what is the expenditure in 2014?",
        location=None,
        external_knowledge_sources=[],
    )

    state_update = retrieve_from_knowledge_base(
        agent_state, config.to_runnable_config()
    )

    documents_content = "\n".join(
        [doc.page_content for doc in state_update.get("documents")]
    )

    assert (
        documents_content
        == """# Average expenditures 2014-2015
Year: 2014, Expenditure: $869.47, Change: 3.4%
Year: 2015, Expenditure: $896.66, Change: 3.1%

# Average expenditures 2015-2016
Year: 2014, Expenditure: $869.47, Change: 3.4%
Year: 2015, Expenditure: $896.66, Change: 3.1%
"""
    )


@pytest.mark.vcr
def test_generate_from_knowledge_base(config, travel_knowledge_documents, snapshot):
    agent_state = AgentState(
        messages=[HumanMessage(content="what are some nice things to see in Paris?")],
        documents=travel_knowledge_documents[0:1],
        query="what are some nice things to see in Paris?",
        location=None,
        external_knowledge_sources=[],
    )

    state_update = generate(agent_state, config.to_runnable_config())

    assert state_update["messages"][-1].content == snapshot


@pytest.mark.vcr
def test_generate_from_external_service(config, snapshot):
    agent_state = AgentState(
        messages=[HumanMessage(content="what is the weather like in paris?")],
        documents=[
            Document(
                page_content="Current weather information for paris: Temperature: -0.55°C, Feels like: -3.75°C, Humidity: 95%, Status: mist",
                id="external[weather]",
            )
        ],
        query="what is the weather like in paris?",
        location=None,
        external_knowledge_sources=[],
    )

    state_update = generate(agent_state, config.to_runnable_config())

    assert state_update["messages"][-1].content == snapshot


@pytest.mark.vcr
def test_graph_with_weather_query(config, travel_knowledge_documents, snapshot):
    vector_store = services.get_vector_store(config)
    vector_store.add_documents(travel_knowledge_documents)
    agent_state = AgentState(
        messages=[HumanMessage(content="what is the weather like in paris?")],
        documents=[],
        query=None,
        location=None,
        external_knowledge_sources=[],
    )

    response = get_graph().invoke(agent_state, config.to_runnable_config())

    assert response.get("messages", [])[-1].content == snapshot


@pytest.mark.vcr
def test_graph_with_knowledge_query(config, travel_knowledge_documents, snapshot):
    vector_store = services.get_vector_store(config)
    vector_store.add_documents(travel_knowledge_documents)
    agent_state = AgentState(
        messages=[HumanMessage(content="what are some nice things to see in Paris?")],
        documents=[],
        query=None,
        location=None,
        external_knowledge_sources=[],
    )

    response = get_graph().invoke(agent_state, config.to_runnable_config())

    assert response.get("messages", [])[-1].content == snapshot


@pytest.mark.parametrize(
    "user_query",
    [
        "What's the trend in auto insurance costs over the last 3 years?",
        # "Compare insurance costs between different regions",
        # "What factors most influence insurance costs?",
        # "Generate a summary of key findings from the data",
    ],
)
@pytest.mark.vcr
def test_graph_with_insurance_queries(
    config, user_query, insurance_data_documents, snapshot
):
    vector_store = services.get_vector_store(config)
    vector_store.add_documents(insurance_data_documents)

    agent_state = AgentState(
        messages=[HumanMessage(content=user_query)],
        documents=[],
        query=None,
        location=None,
        external_knowledge_sources=[],
    )

    response = get_graph().invoke(agent_state, config.to_runnable_config())

    assert response.get("messages", [])[-1].content == snapshot


@pytest.mark.xfail(reason="node not used")
@pytest.mark.vcr
def test_reformulate_query__STUB(config):
    agent_state = AgentState(
        messages=[
            HumanMessage(
                content="What's the trend in auto insurance costs from year 2012 to year 2014?",
            )
        ],
        documents=[],
        query=None,
        location=None,
        external_knowledge_sources=[],
    )

    state_update = reformulate_query(agent_state, config.to_runnable_config())

    assert state_update == {"query": "Auto insurance cost trend 2012-2014"}
