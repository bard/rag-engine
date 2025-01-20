import pytest
from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage

from src.config import AgentConfig
from src.query import AgentState, get_graph, retrieve, reformulate_query, classify_query
from src import services


@pytest.mark.vcr
def test_reformulate_query(agent_config):
    agent_state = AgentState(
        messages=[
            HumanMessage(
                content="what are the year-to-year changes between 2012 and 2014"
            )
        ],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query=None,
        location=None,
    )

    state_update = reformulate_query(agent_state, agent_config)

    assert state_update == {"query": '"year-to-year changes 2012-2014"'}


@pytest.mark.vcr
def test_classify_query(agent_config):
    agent_state = AgentState(
        messages=[HumanMessage(content="how is the weather in Paris?")],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query=None,
        location=None,
    )

    state_update = classify_query(agent_state, agent_config)

    assert state_update == {
        "is_weather_query": True,
        "location": "Paris",
        "query": "how is the weather in Paris?",
    }


# TODO rename
def test_retrieve_2(agent_config, travel_info_documents, snapshot):
    vector_store = services.get_vector_store(
        AgentConfig.from_runnable_config(agent_config)
    )
    vector_store.add_documents(travel_info_documents)

    agent_state = AgentState(
        messages=[
            HumanMessage(content="o"),
        ],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query="what are some nice things to see in Paris?",
        location=None,
    )

    state_update = retrieve(agent_state, agent_config)

    assert state_update.get("documents") == snapshot


def test_retrieve(agent_config):
    vector_store = services.get_vector_store(
        AgentConfig.from_runnable_config(agent_config)
    )
    vector_store.add_texts(
        [
            "Year: 2012, Expenditure: $812.40, Change: 2.2%",
            "Year: 2013, Expenditure: $841.06, Change: 3.5%",
            "Year: 2014, Expenditure: $869.47, Change: 3.4%",
            "Year: 2015, Expenditure: $896.66, Change: 3.1%",
        ],
        metadatas=[{"year": 2012}, {"year": 2013}, {"year": 2014}, {"year": 2015}],
    )
    agent_state = AgentState(
        messages=[
            HumanMessage(content="what is the expenditure in 2014?"),
        ],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query="what is the expenditure in 2014?",
        location=None,
    )

    state_update = retrieve(agent_state, agent_config)

    documents_content = "\n".join(
        [doc.page_content for doc in state_update.get("documents")]
    )

    assert (
        documents_content
        == """Year: 2014, Expenditure: $869.47, Change: 3.4%
Year: 2013, Expenditure: $841.06, Change: 3.5%"""
    )


@pytest.mark.vcr
def test_graph_with_weather_query(agent_config, travel_info_documents, snapshot):
    vector_store = services.get_vector_store(
        AgentConfig.from_runnable_config(agent_config)
    )
    vector_store.add_documents(travel_info_documents)
    agent_state = AgentState(
        messages=[
            HumanMessage(content="what is the weather like in paris?"),
        ],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query=None,
        location=None,
    )

    response = get_graph().invoke(agent_state, agent_config)

    assert response.get("messages", [])[-1].content == snapshot


@pytest.mark.vcr
def test_graph_with_travel_info_query(agent_config, travel_info_documents, snapshot):
    vector_store = services.get_vector_store(
        AgentConfig.from_runnable_config(agent_config)
    )
    vector_store.add_documents(travel_info_documents)
    agent_state = AgentState(
        messages=[
            HumanMessage(content="what are some nice things to see in Paris?"),
        ],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query=None,
        location=None,
    )

    response = get_graph().invoke(agent_state, agent_config)

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
    agent_config, user_query, insurance_data_documents, snapshot
):
    vector_store = services.get_vector_store(
        AgentConfig.from_runnable_config(agent_config)
    )
    vector_store.add_documents(insurance_data_documents)

    agent_state = AgentState(
        messages=[
            HumanMessage(
                content=user_query,
            ),
        ],
        documents=[],
        weather_info=None,
        is_weather_query=False,
        query=None,
        location=None,
    )

    response = get_graph().invoke(agent_state, agent_config)

    assert response.get("messages", [])[-1].content == snapshot
