import pprint
import pytest
from langchain.schema import HumanMessage
from langchain_core.messages import HumanMessage

from src import services
from src.workflow_query import (
    AgentState,
    get_graph,
    retrieve_from_knowledge_base,
)


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
        metadatas=[
            {"topic_id": "UNCATEGORIZED"},
            {"topic_id": "UNCATEGORIZED"},
            {"topic_id": "UNCATEGORIZED"},
        ],
    )
    agent_state = AgentState(
        messages=[
            HumanMessage(content="what is the expenditure in 2014?"),
        ],
        retrieved_knowledge=[],
        query="what is the expenditure in 2014?",
        topic_id=None,
        external_knowledge_sources=[],
    )

    state_update = retrieve_from_knowledge_base(
        agent_state, config.to_runnable_config()
    )

    documents_content = "\n".join(
        [doc.page_content for doc in state_update["retrieved_knowledge"]]
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
@pytest.mark.parametrize(
    "user_query",
    [
        "What is the trend in auto insurance costs between 2012 and 2015?",
        # "Compare insurance costs between different regions",
        # "What factors most influence insurance costs?",
        # "Generate a summary of key findings from the data",
    ],
)
def test_graph_with_insurance_queries(
    config, user_query, insurance_data_documents, snapshot
):
    services.get_vector_store(config).add_documents(insurance_data_documents)

    agent_state = AgentState(
        messages=[HumanMessage(content=user_query)],
        retrieved_knowledge=[],
        query=None,
        topic_id=None,
        external_knowledge_sources=[],
    )

    response = get_graph().invoke(agent_state, config.to_runnable_config())

    assert response.get("messages", [])[-1].content == snapshot
