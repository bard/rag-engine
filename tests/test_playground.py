import pytest
import base64
from src.agents import VectorStore
from src.settings import Settings


@pytest.mark.skip
def test_load_insurance_data_with_data_url(sample_html):
    settings = Settings.model_validate({})
    encoded_html = base64.b64encode(sample_html.encode("utf-8")).decode("utf-8")
    data_url = f"data:text/html;base64,{encoded_html}"
    documents = load_insurance_data_from_url(data_url)

    vector_store = VectorStore(
        pinecone_api_key=settings.pinecone_api_key,
        pinecone_index_name="ateam",
        openai_api_key=settings.openai_api_key,
    )
    vector_store.add_documents(documents)
