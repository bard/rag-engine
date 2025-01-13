from langchain_chroma.vectorstores import Chroma
from sqlalchemy.orm import Session
from src.data import SQLInsuranceRecord
from sqlalchemy import create_engine
from langchain_core.runnables.config import RunnableConfig
from src.agents.ingest import ingest, AgentState
from sqlalchemy import create_engine


# @pytest.mark.integration
def test_ingest_node(sample_html_data_url, tmp_db_url, snapshot, tmp_path):
    config = RunnableConfig(
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
    state = AgentState(url=sample_html_data_url)

    new_agent_state = ingest(state, config)

    database_state = Session(create_engine(tmp_db_url)).query(SQLInsuranceRecord).all()

    vector_store_state = Chroma(
        collection_name="documents", persist_directory=f"{tmp_path}/chroma"
    ).get()

    assert new_agent_state == snapshot
    assert database_state == snapshot
    assert vector_store_state == snapshot
