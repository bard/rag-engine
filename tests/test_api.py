import pprint
import pytest
from sqlalchemy.orm import Session

from src import services
from src.db import SqlKnowledgeBaseDocument, SqlTopic


def test_healthcheck(api_client):
    response = api_client.get("/healthcheck")

    assert response.status_code == 200


@pytest.mark.vcr
@pytest.mark.integration
def test_poor_mans_e2e_test(api_client, config, snapshot):
    # create and delete topics

    response = api_client.post("/topics", json={"name": "London"})
    assert response.status_code == 200
    london_topic_id = response.json()["id"]
    assert len(london_topic_id) == 36  # UUID format length

    response = api_client.post("/topics", json={"name": "    london "})
    assert response.status_code == 409
    assert response.json() == {"detail": "Topic 'london' already exists"}

    response = api_client.delete(f"/topics/{london_topic_id}")
    assert response.status_code == 200

    response = api_client.post("/topics", json={"name": "Paris"})
    paris_topic_id = response.json()["id"]
    with Session(services.get_db(config)) as session:
        stored_topics = session.query(SqlTopic).all()
        assert len(stored_topics) == 1
        assert stored_topics[0].name == "Paris"

    # create note in topic

    response = api_client.post(
        "/notes",
        json={
            "content": "Paris, the 'City of Light,' boasts iconic landmarks such as the Eiffel Tower, offering panoramic views from its observation decks. The Louvre Museum, home to masterpieces like the Mona Lisa, attracts art enthusiasts worldwide. Notre-Dame Cathedral, a Gothic architectural marvel, stands on the Île de la Cité. The Champs-Élysées, lined with shops and cafes, leads to the Arc de Triomphe, honoring those who fought for France. Montmartre, with its artistic heritage, features the Basilica of the Sacré-Cœur atop its hill.",
            "content_type": "text/plain",
            "topic_id": paris_topic_id,
        },
    )

    # verify that note is in the knowledge base

    assert response.status_code == 200
    note_id = response.json()["id"]
    assert isinstance(note_id, str)
    with Session(services.get_db(config)) as session:
        stored_notes = session.query(SqlKnowledgeBaseDocument).all()
        assert len(stored_notes) == 1
        assert "Paris, the 'City of Light,' boasts iconic" in stored_notes[0].content

    # Verify note exists in vector store
    vector_store = services.get_vector_store(config)
    results = vector_store.similarity_search("Paris landmarks", k=1)
    assert len(results) > 0
    assert "Paris" in results[0].page_content
    assert results[0].metadata["source_id"] == note_id

    # query

    response = api_client.get(
        "/query",
        params={
            "q": "What are the good things to see?",
            "topic_id": paris_topic_id,
        },
    )

    assert response.status_code == 200
    answer = response.json()["answer"]

    # The answer should mention at least some of these landmarks
    assert any(
        landmark in answer
        for landmark in [
            "Eiffel Tower",
            "Louvre",
            "Notre-Dame",
            "Champs-Élysées",
            "Arc de Triomphe",
        ]
    )

    # run query requiring external knowledge

    response = api_client.get(
        "/query",
        params={
            "q": "What is the current weather?",
            "topic_id": paris_topic_id,
        },
    )

    assert response.status_code == 200
    assert response.json() == snapshot

    # delete note

    response = api_client.delete(f"/notes/{note_id}")
    assert response.status_code == 200
    assert response.json() == {"message": f"Note {note_id} deleted successfully"}

    with Session(services.get_db(config)) as session:
        stored_notes = session.query(SqlKnowledgeBaseDocument).all()
        assert len(stored_notes) == 0
